# system-related imports
import sys
import tempfile
import gc
import tqdm
import os
import shutil
from pathlib import Path
from natsort import natsorted
from ast import literal_eval

# computer vision-related imports
import cv2
import torch
import supervision as sv
import numpy as np

# model import
from sam2.build_sam import build_sam2
    # hydra already initialized fix
from hydra import initialize
import hydra
hydra.core.global_hydra.GlobalHydra.instance().clear()
    # end fix, import the rest of the model
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2_camera_predictor
import pickle


# typing import
from typing import Any, Union, List, Dict
from cv2.typing import MatLike
from numpy import ndarray
from gradio import Progress as gr_progress


def union_binary_masks(masks, hw: tuple[int, int] | None = None) -> np.ndarray:
    """
    Fast union (OR) of a list of binary masks.
    - masks: iterable of HxW arrays (bool or 0/1), possibly mixed shapes
    - hw: (H, W) target size if masks may differ in size; uses nearest-neighbor resize
    Returns: HxW bool array
    """
    # filter out Nones and ensure list
    masks = [m for m in masks if m is not None]
    if not masks:
        return np.zeros((1, 1), dtype=bool) if hw is None else np.zeros(hw, dtype=bool)

    # If all shapes match, take the vectorized fast path (no Python loop)
    shapes = {m.shape[:2] for m in masks}
    if len(shapes) == 1:
        # coerce to bool without copy when possible
        stack = np.stack([np.asarray(m, dtype=bool, order="C") for m in masks], axis=0)
        return np.any(stack, axis=0)

    # Shapes differ: require target size
    if hw is None:
        raise ValueError("Masks have different shapes; pass hw=(H, W) to resize.")
    H, W = hw
    out = np.zeros((H, W), dtype=bool)

    # Streamed OR to avoid big temporary stacks
    for m in masks:
        mm = np.asarray(m)
        if mm.dtype != np.bool_:
            # treat nonzero as True
            mm = mm != 0
        if mm.shape[:2] != (H, W):
            # resize with nearest to keep binary nature
            mm = cv2.resize(mm.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        out |= mm
    return out

# segment-anything model class
class SegmentAnything2:
    def __init__(self, model_type: str = "base_plus", batch_num: int = 30) -> None:
        """ Initialize the SAM model """
        
        # set the device
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # set the model type
        allowed_types = ["tiny", "small", "base_plus", "large"]
        MODEL_TYPE = model_type.lower()
        if MODEL_TYPE not in allowed_types:
            print(f"Model type '{model_type}' is not allowed")
            MODEL_TYPE = "base_plus"
        
        # set the weights path
        CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / f"sam2_hiera_{MODEL_TYPE}.pt"
        
        configs = {
            "tiny": "t",
            "small": "s",
            "base_plus": "b+",
            "large": "l"
        }
        
        MODEL_CFG = f"sam2_hiera_{configs[MODEL_TYPE]}.yaml"
        
        # initialize and  load the model
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        cfg_dir = "sam2"   # wherever your YAMLs are
        initialize(config_path=str(cfg_dir), version_base="1.2", job_name="sam2_build")

        sam2_model = build_sam2(MODEL_CFG, CHECKPOINT_PATH.as_posix())
        self.model = sam2_model
        
        self.IMG_BATCH_SIZE = batch_num
        
        # initialize the mask generator and annotator
        POINTS_PER_BATCH = 16 # 64 is default, any other number, lower the memory usage
        self.img_predictor = SAM2ImagePredictor(sam2_model)
        self.video_predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT_PATH.as_posix(), fill_hole_area=1)
        self.camera_predictor = build_sam2_camera_predictor(MODEL_CFG, CHECKPOINT_PATH.as_posix())
        self.mask_generator = SAM2AutomaticMaskGenerator(sam2_model, points_per_batch=POINTS_PER_BATCH)
        self.mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        self.track_mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.TRACK)
        
        # make sure to have a spare folder for the video processing
        self.video_processing_folder = Path(__file__).parent / "tempfiles"
        self.video_processing_folder.mkdir(exist_ok=True)
        self._clear_processing_folder()
        
        # set the video suffix and standard fourcc for video creation
        self.video_suffix = '.mp4'
        self.standard_fourcc = 'mp4v'
    
    def get_model_name(self) -> str:
        """ Return the model name """
        return "Segment-Anything"
    
    def _clear_processing_folder(self):
        """ Clear the video processing folder """
        # loop through the files in the video processing folder
        for file in self.video_processing_folder.iterdir():
            try:
                if file.is_file(): # if the file is a file, delete it
                    os.remove(file.as_posix())
                elif file.is_dir(): # if the file is a directory, delete it as a tree
                    shutil.rmtree(file.as_posix())
            except Exception as e: # if an error occurs, print the error, then continue
                print(f"Failed to delete '{file}', error: {e}")
                continue
    
    def img_shot(self,
                    image: Any) -> List[Dict[str, Any]]:
        """ Get the masks of an image """
        return self.mask_generator.generate(image)
    
    def _get_color(self):
        """ Get the color of the mask """
        random_r = np.random.choice(range(40, 210))
        random_g = np.random.choice(range(40, 210))
        random_b = np.random.choice(range(40, 210))
        return random_b, random_g, random_r
    
    def _folder_zero_shot(self,
                         folder_path: Path,):
        """ Zero-shot of the model on a folder """
        # Zero-shot of the model
        frame_list = natsorted(list(folder_path.iterdir())) # list of frames
        frame_0 = cv2.imread(frame_list[0].as_posix()) # read the first frame
        mask_res = self.img_shot(frame_0) # get the masks from the first frame
        labels = [[1] for i in range(len(mask_res))] # clicks, [1] for positive, [0] for negative
        ann_frame_idx = 0 # annotation frame index, 0 for beggining
        colors = [[self._get_color()] for i in range(len(mask_res))] # random colors for the objects
        
        inference_state = self.video_predictor.init_state(video_path = folder_path.as_posix())
        for mask_idx, mask_res in enumerate (mask_res):
            _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points(
                inference_state = inference_state,
                frame_idx = ann_frame_idx,
                obj_id = mask_idx,
                points = mask_res["point_coords"],
                labels = labels[mask_idx]
            )
        return colors, frame_list, inference_state
    
    def _set_manual_track(self,
                         track_coords: List[Dict[str, Any]],
                         folder_path: Path,):
        
        # Zero-shot of the model
        frame_list = natsorted(list(folder_path.iterdir())) # list of frames
        frame_0 = cv2.imread(frame_list[0].as_posix()) # read the first frame
        point_coords_list = [[[coord["x"], coord["y"]]] for coord in track_coords]
        labels = [[1] for i in range(len(point_coords_list))] # clicks, [1] for positive, [0] for negative
        ann_frame_idx = 0 # annotation frame index, 0 for beggining
        colors = [[self._get_color()] for i in range(len(point_coords_list))] # random colors for the objects
        
        inference_state = self.video_predictor.init_state(video_path = folder_path.as_posix())
        for point_idx, point_coords in enumerate(point_coords_list):
            _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points(
                inference_state = inference_state,
                frame_idx = ann_frame_idx,
                obj_id = point_idx,
                points = point_coords,
                labels = labels[point_idx]
            )
        return colors, frame_list, inference_state
    
    def process_image(self,
                      image: Any,
                      **kwargs
                      )-> Any:
        
        """
        Process an image with the SAM model
        
        <h3> Inputs </h3>
        <b> > image: </b><br> Image instance <br>
        
        <h3> Outputs </h3>
        <b> > annotated_image: </b><br> Annotated image instance <br>
        """
        
        if image is None:
            return None

        # generate masks with SAM2
        masks = self.img_shot(image)
            
        # convert masks to detections
        detections = sv.Detections.from_sam(masks)
        
        # annotate the image with the detections
        annotated_image = self.mask_annotator.annotate(image, detections)
        
        return annotated_image
    
    def segment_single_click(
        self,
        image: Any,
        click: Dict[str, int],
        multimask_output: bool = True
    ) -> np.ndarray:
        """
        Segment a single object in an image given one positive click.

        Args:
            image:        H×W×C numpy array (BGR).
            click:        {"x": int, "y": int} in pixel coordinates.
            multimask_output: if True, SAM returns multiple candidate masks.

        Returns:
            H×W boolean mask of the best segment (dtype=bool).
        """
        # 1. Embed the image
        self.img_predictor.set_image(image)

        # 2. Prepare the click prompt
        coords = np.array([[click["x"], click["y"]]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)  # 1 = foreground

        # 3. Predict masks + IOU scores
        masks, ious, _ = self.img_predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=multimask_output,
            return_logits=False
        )
        

        # masks is an array of shape [C,H,W] with boolean or 0/1 floats
        # 4. Pick the best mask by highest IOU
        best_idx = int(np.argmax(ious))
        best_mask = masks[best_idx].astype(bool)  # H×W, dtype=bool

        return best_mask
    
    def segment_multiple_clicks(
        self,
        image: Any,
        clicks: List[Dict[str, int]],
        multimask_output: bool = True
    ) -> List[np.ndarray]:
        """
        Segment one object per click, picking the highest‐IOU mask for each.

        Args:
            image:           H×W×C numpy array (BGR or RGB).
            clicks:          list of {"x":int,"y":int, optional "label":0|1}
                             label defaults to 1 (foreground).
            multimask_output: if True, SAM returns multiple masks per click.

        Returns:
            List of H×W boolean masks, one per click.
        """
        # 1) Compute embeddings once
        self.img_predictor.set_image(image)

        masks_out = []
        for ck in clicks:
            # 2) build prompt
            coords = np.array([[ck["x"], ck["y"]]], dtype=np.float32)
            labels = np.array([ck.get("label", 1)], dtype=np.int32)

            # 3) predict
            masks, ious, _ = self.img_predictor.predict(
                point_coords=coords,
                point_labels=labels,
                multimask_output=multimask_output,
                return_logits=False
            )  # masks.shape == [C, H, W], ious.shape == [C]

            # 4) pick the mask with highest predicted IOU
            best_idx  = int(np.argmax(ious))
            best_mask = masks[best_idx].astype(bool)

            masks_out.append(best_mask)

        return masks_out
    
    def segment_with_clicks(
        self,
        image: Any,
        clicks: List[Dict[str, int]],
        multimask_output: bool = False
    ) -> np.ndarray:
        """
        Segment a single object given multiple point prompts.

        Args:
            image:            H×W×C array (BGR or RGB).
            clicks:           list of {"x":int,"y":int, optional "label":0|1}.
                              label=1 is foreground, 0 is background.
            multimask_output: if True, SAM returns multiple candidate masks.

        Returns:
            H×W boolean mask for the object (chosen by highest IOU).
        """
        # 1. Compute image embeddings once
        self.img_predictor.set_image(image)

        # 2. Build point arrays
        coords = np.array([[c["x"], c["y"]] for c in clicks], dtype=np.float32)
        labels = np.array([c.get("label", 1) for c in clicks], dtype=np.int32)

        # 3. Predict masks + IOU scores in one shot
        masks, ious, _ = self.img_predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=multimask_output,
            return_logits=False
        )  # masks.shape == [C, H, W], ious.shape == [C]

        # 4. Pick the best mask by highest IOU
        best_idx  = int(np.argmax(ious))
        best_mask = masks[best_idx].astype(bool)  # H×W boolean

        return best_mask
    
    
    def segment_with_clicks_vpred(
        self,
        folder_path: Path,
        clicks: List[Dict[str, int]],
        frame_idx: int = 0,
        obj_id: int = 0,
        multimask_output: bool = False,  # kept for API parity; not used by video predictor
    ) -> np.ndarray:
        """
        Zero-shot on a folder of frames using SAM2VideoPredictor with pos/neg clicks.
        Returns an HxW boolean mask for the specified frame and object.

        Args:
            folder_path: path to a directory of frames (e.g., 00000.jpg, 00001.jpg, ...).
            clicks:      list of {"x": int, "y": int, optional "label": 0|1}.
                        label defaults to 1 (foreground).
            frame_idx:   which frame to segment (default 0).
            obj_id:      your object id (default 0).
        """

        # 1) Initialize a video session from the frames directory
        state = self.video_predictor.init_state(video_path=str(folder_path))

        try:
            # 2) Build prompt tensors (1 = FG, 0 = BG)
            pts = np.asarray([[c["x"], c["y"]] for c in clicks], dtype=np.float32)
            lbs = np.asarray([c.get("label", 1) for c in clicks], dtype=np.int32)
            if pts.shape[0] == 0:
                raise ValueError("No clicks provided.")

            # 3) Add clicks on `frame_idx` and get the output on that same frame
            with torch.inference_mode():
                f_idx, obj_ids, video_res_masks = self.video_predictor.add_new_points(
                    inference_state=state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=pts,
                    labels=lbs,
                    # normalize_coords=True by default; pixel coords are OK
                )

            if obj_id not in obj_ids:
                raise RuntimeError(f"obj_id {obj_id} not in returned object IDs: {obj_ids}")

            k = obj_ids.index(obj_id)
            # `video_res_masks` are logits at video resolution; threshold at 0
            mask = (video_res_masks[k, 0] > 0.0).detach().cpu().numpy().astype(bool)
            return mask
        
        finally:
            self.video_predictor.reset_state(state)


           
    def process_video_batches(self,
                      video_path: str,
                      input_folder: Path,
                      output_folder: Path,
                      progress_bar: Union[gr_progress, None] = None
                    ) -> str:
        
        """
        Process a video with the SAM2 model in batches
        
        <h3> Inputs </h3>
        <b> > video_path: </b><br> Path to the video <br>
        <b> > progress_bar: </b><br> Progress bar instance, tracks tqdm <br>
        
        <h3> Outputs </h3>
        <b> > output_path: </b><br> Path to the output video <br>
        """
        
        # start reading the video file
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_name_len = 10
        
        # read and save the frames
        img_count = 0
        batch_no = 0
        while cap.isOpened():
            # read the frame
            ret, frame = cap.read()
            if not ret: # if the frame is not read, break
                break
            
            # check the corresponding batch folder
            batch_no = img_count // self.IMG_BATCH_SIZE
            batch_path = input_folder / f"batch_{batch_no}"
            if not batch_path.exists():
                batch_path.mkdir()
                batch_no += 1
                
            # get the frame position
            position = cap.get(cv2.CAP_PROP_POS_FRAMES)
            # create the frame name in the format "000001.jpg"
            frame_name = str(int(position)).zfill(frame_name_len)
            # save the frame
            frame_path = batch_path / f"{frame_name}.jpg"
            cv2.imwrite(frame_path.as_posix(), frame)
            # increment the image count
            img_count += 1
        cap.release() # release the video capture
        
        # initialize the video recorder
        output_video = output_folder / f"output{self.video_suffix}"
        fourcc = cv2.VideoWriter_fourcc(*self.standard_fourcc)
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        batch_list = natsorted(input_folder.iterdir())
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # loop through the video batches
            for batch_folder in tqdm.tqdm(batch_list,
                                        desc="Processing video batches"):
                if batch_folder.is_dir():
                    # perform the zero-shot on the batch folder
                    colors, frame_list, inference_state = self._folder_zero_shot(batch_folder)
                    # loop through the frames
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                        # read the frame
                        frame_path = frame_list[out_frame_idx]
                        frame = cv2.imread(frame_path.as_posix())
                        colored_masks = np.array(frame, copy=True, dtype=np.uint8)
                        # loop through the tracked objects
                        for idx, obj_id in enumerate(out_obj_ids):
                            # get the mask
                            out_mask = (out_mask_logits[idx] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                                np.uint8
                            ) * 255
                            # get the color of the mask
                            mask_color = np.array(colors[obj_id], dtype=np.uint8) 
                            # create new frame with the mask area filled with color
                            colored_masks[out_mask.squeeze() > 0] = mask_color
                            # add the colored mask to the frame
                            contours, _ = cv2.findContours(out_mask.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            frame = cv2.drawContours(frame, contours, -1, (0,0,0), thickness=3)
                            
                        frame = cv2.addWeighted(frame, 0.75, colored_masks, 0.25, 0)
                        # write the frame to the output video
                        out.write(frame)
                        torch.cuda.empty_cache()
                    self.video_predictor.reset_state(inference_state)
        # release the video writer
        out.release()
        return output_video.as_posix()
    
    def process_video_manual_track(self,
                                   video_path: str,
                                   track_coords: List[Dict[str, Any]],
                                   input_folder: Path,
                                   output_folder: Path,
                                   progress_bar: Union[gr_progress, None] = None) -> str:
        # start reading the video file
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_name_len = 10
        
        # read and save the frames
        while cap.isOpened():
            # read the frame
            ret, frame = cap.read()
            if not ret: # if the frame is not read, break
                break
            # get the frame position
            position = cap.get(cv2.CAP_PROP_POS_FRAMES)
            # create the frame name in the format "000001.jpg"
            frame_name = str(int(position)).zfill(frame_name_len)
            # save the frame
            frame_path = input_folder / f"{frame_name}.jpg"
            cv2.imwrite(frame_path.as_posix(), frame)
        cap.release() # release the video capture
        
        # initialize the video recorder
        output_video = output_folder / f"output{self.video_suffix}"
        fourcc = cv2.VideoWriter_fourcc(*self.standard_fourcc)
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        colors, frame_list, inference_state = self._set_manual_track(track_coords, input_folder)
        # start the tracking
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                # read the frame
                frame_path = frame_list[out_frame_idx]
                frame = cv2.imread(frame_path.as_posix())
                colored_masks = np.array(frame, copy=True, dtype=np.uint8)
                # loop through the tracked objects
                for idx, obj_id in enumerate(out_obj_ids):
                    # get the mask
                    out_mask = (out_mask_logits[idx] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                        np.uint8
                    ) * 255
                    # get the color of the mask
                    mask_color = np.array(colors[obj_id], dtype=np.uint8) 
                    # create new frame with the mask area filled with color
                    colored_masks[out_mask.squeeze() > 0] = mask_color
                    # add the colored mask to the frame
                    contours, _ = cv2.findContours(out_mask.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    frame = cv2.drawContours(frame, contours, -1, (0,0,0), thickness=3)
                    
                frame = cv2.addWeighted(frame, 0.75, colored_masks, 0.25, 0)
                # write the frame to the output video
                out.write(frame)
                torch.cuda.empty_cache()
            self.video_predictor.reset_state(inference_state)
        # release the video writer
        out.release()
        return output_video.as_posix()
    
    def process_video(self,
                      video_path: str,
                      dropdown_opt: str = None,
                      track_coords: str = None,
                      progress_bar: Union[gr_progress, None] = None) -> str:
        
        """Process a video with the SAM2 model"""
        
        print(f"Video path: {video_path}")
        if video_path is None:
            pass
        self._clear_processing_folder()
        
        # create the input folder
        input_folder = self.video_processing_folder / "input"
        input_folder.mkdir(exist_ok=True)
        
        # create the output folder
        output_folder = self.video_processing_folder / "output"
        output_folder.mkdir(exist_ok=True)
        
        if dropdown_opt == "Tracking autoupdate":
            output_path = self.process_video_batches(video_path, input_folder, output_folder, progress_bar)
        elif dropdown_opt == "Choose objects":
            track_coords = literal_eval(track_coords)
            output_path = self.process_video_manual_track(video_path, track_coords, input_folder, output_folder, progress_bar)
            
        return output_path
    
    def track_video_single_object(
        self,
        folder_path: Path,
        positive_clicks: List[Dict[str, int]],
        negative_clicks: List[Dict[str, int]],
        frame_idx: int = 0
    ) -> List[np.ndarray]:
        """
        1) Add pos+neg clicks on `frame_idx` as one object (obj_id=0)
        2) Propagate backward to fill [0..frame_idx], then forward to fill [frame_idx..end]
        """

        # init state (assumes folder_path already has 00000.jpg, 00001.jpg…)
        inference_state = self.video_predictor.init_state(video_path=str(folder_path))

        # bundle all clicks into one object
        pts = [[c["x"], c["y"]] for c in positive_clicks] \
            + [[c["x"], c["y"]] for c in negative_clicks]
        lbs = [1]*len(positive_clicks) + [0]*len(negative_clicks)

        # register your single object on the chosen frame
        self.video_predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=0,
            points=pts,
            labels=lbs,
        )

        # pre-allocate per-frame masks so we can do two passes safely
        num_frames = inference_state["num_frames"]
        masks: List[np.ndarray] = [None] * num_frames

        # ---- Pass 1: backward from the clicked frame (fills ... frame_idx-1, ..., 0)
        for t, obj_ids, mask_logits in self.video_predictor.propagate_in_video(
            inference_state,
            start_frame_idx=frame_idx,
            reverse=True
        ):
            if 0 in obj_ids:
                idx0 = obj_ids.index(0)
                logit = mask_logits[idx0]  # Tensor (1×H×W)
                mask = (logit
                        .permute(1, 2, 0)   # H×W×1
                        .cpu()
                        .numpy()
                        .squeeze() > 0
                    ).astype(np.uint8)
                masks[t] = mask

        # ---- Pass 2: forward from the clicked frame (fills frame_idx, frame_idx+1, ..., end)
        for t, obj_ids, mask_logits in self.video_predictor.propagate_in_video(
            inference_state,
            start_frame_idx=frame_idx,
            reverse=False
        ):
            if 0 in obj_ids:
                idx0 = obj_ids.index(0)
                logit = mask_logits[idx0]  # Tensor (1×H×W)
                mask = (logit
                        .permute(1, 2, 0)   # H×W×1
                        .cpu()
                        .numpy()
                        .squeeze() > 0
                    ).astype(np.uint8)
                masks[t] = mask

        # cleanup so predictor can be reused
        self.video_predictor.reset_state(inference_state)

        # fill any missing frames (shouldn't happen, but just in case)
        any_mask = next((m for m in masks if m is not None), None)
        if any_mask is None:
            # infer shape from first frame on disk
            frame_files = [p for p in natsorted(Path(folder_path).iterdir())
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
            if frame_files:
                H, W = cv2.imread(str(frame_files[0])).shape[:2]
                any_mask = np.zeros((H, W), dtype=np.uint8)
            else:
                any_mask = np.zeros((1, 1), dtype=np.uint8)

        return [m if m is not None else np.zeros_like(any_mask, dtype=np.uint8)
                for m in masks]

    
    def init_camera(self,
                    click_coords: List[Dict[str, Any]],
                    frame: Any,
                    return_masks = False) -> None:
        """ Initialize the camera """
        # load the first frame
        self.camera_predictor.load_first_frame(frame)
        # create the initial annotations
        point_coords_list = [[[coord["x"], coord["y"]]] for coord in click_coords]
        labels = [[1] for i in range(len(point_coords_list))] # clicks, [1] for positive, [0] for negative
        ann_frame_idx = 0 # annotation frame index, 0 for beggining
        self.camera_colors = [[self._get_color()] for i in range(len(point_coords_list))] # random colors for the objects
        
        for point_idx, point_coords in enumerate(point_coords_list):
            _, out_obj_ids, out_mask_logits = self.camera_predictor.add_new_points(
                frame_idx = ann_frame_idx,
                obj_id = point_idx,
                points = point_coords,
                labels = labels[point_idx]
            )
            
        # return objects ids and logits
        if return_masks:
            return out_obj_ids, [self._get_binary_mask(logit) for logit in out_mask_logits]
        return out_obj_ids, out_mask_logits

    def _propagate_camera(self,
                          frame: Any) -> Any:
        """ Propagate the camera, returns ids and logits """
        out_obj_ids, out_mask_logits = self.camera_predictor.track(frame)
        return out_obj_ids, out_mask_logits
    
    def _get_binary_mask(self, logits):
        return (logits > 0.0).permute(1, 2, 0).cpu().numpy().astype(bool)

    def propagate_binary(self, frame: Any) -> Any:
        """ Propagate the mask """
        out_obj_ids, out_mask_logits = self._propagate_camera(frame)
        return out_obj_ids, [self._get_binary_mask(logit) for logit in out_mask_logits]
    
    def _resize_bool_mask(mask: np.ndarray, HW: tuple[int, int]) -> np.ndarray:
        H, W = HW
        m = mask.astype(np.uint8)
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        return (m > 0)
    
    def propagate_binary_multi(
        self,
        frame: Any,
        keep_ids: list[int] | None = None,
    ) -> dict[int, np.ndarray]:
        """
        Returns {obj_id: HxW bool mask} for the current frame.
        If keep_ids is given, filters to that subset.
        """
        out_obj_ids, bin_masks = self.propagate_binary(frame)
        out = {}
        for oid, m in zip(out_obj_ids, bin_masks):
            oid = int(oid)
            if (keep_ids is None) or (oid in keep_ids):
                out[oid] = m.astype(bool)
        return out

    def union_masks(self, masks_dict: dict[int, np.ndarray], ids: list[int] | None = None) -> np.ndarray:
        """
        Union selected masks -> HxW bool. If ids is None, use all.
        """
        if not masks_dict:
            return np.zeros((1, 1), dtype=bool)
        ids = list(masks_dict.keys()) if ids is None else ids
        base = None
        for oid in ids:
            if oid not in masks_dict: 
                continue
            m = masks_dict[oid].astype(bool)
            base = m if base is None else (base | m)
        return base if base is not None else np.zeros((1, 1), dtype=bool)

    def propagate_union_mask(
        self,
        frame: Any,
        keep_ids: list[int] | None = None,
    ) -> np.ndarray:
        md = self.propagate_binary_multi(frame, keep_ids=keep_ids)
        return self.union_masks(md)


    def _visualize_propagation(self,
                               frame: Any,
                               out_obj_ids: List[int],
                               out_mask_logits: List[torch.Tensor]) -> MatLike:
        """ Visualize the propagation """
        colored_masks = np.array(frame, copy=True, dtype=np.uint8)
        for idx, obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[idx] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
            mask_color = np.array(self.camera_colors[obj_id], dtype=np.uint8)
            colored_masks[out_mask.squeeze() > 0] = mask_color
            contours, _ = cv2.findContours(out_mask.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            frame = cv2.drawContours(frame, contours, -1, (0,0,0), thickness=3)
        frame = cv2.addWeighted(frame, 0.75, colored_masks, 0.25, 0)
        return frame
        
    def track_camera(self,
                     frame: Any) -> MatLike:
        """ Track the camera further """
        # out_obj_ids, out_mask_logits = self.camera_predictor.track(frame)
        out_obj_ids, out_mask_logits = self._propagate_camera(frame)
        colored_frame = self._visualize_propagation(frame, out_obj_ids, out_mask_logits)
        return colored_frame
    
    def propagate_pkl(self,
                         frame: Any) -> Any:
        """ Track the camera further and return the pickled file """
        self._clear_processing_folder()
        pickle_path = self.video_processing_folder / "masks.pkl"
        
        out_obj_ids, out_mask_logits = self._propagate_camera(frame)
        masks = {}
        for out_obj_id, out_mask_logit in zip(out_obj_ids, out_mask_logits):
            out_mask = (out_mask_logit > 0.0).permute(1, 2, 0).cpu().numpy().squeeze()
            masks[out_obj_id] = out_mask
        with open(pickle_path, "wb") as f:
            pickle.dump(masks, f)
        
        visualization = self._visualize_propagation(frame, out_obj_ids, out_mask_logits)
        return visualization, pickle_path.as_posix()

        
        
    def release(self):
        """ Release the model from memory """
        # Clear CUDA cache if used
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("> CUDA cache cleared")
        # Force garbage collection
        gc.collect()
        print("> Garbage collection complete")
        