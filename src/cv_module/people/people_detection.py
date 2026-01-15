from ultralytics import YOLO
from ultralytics.engine.results import Results, Keypoints
from omegaconf import OmegaConf
import numpy as np
import torch
import torchvision
import cv2
import os

from src.cv_module.basic_object import DetectedObject


class YoloDetectorSlicing:
    """
    Detect people with YOLOv8 detection

    Warning: contains hardcoded "levels" logic "faraway object should be at the center of the image" !
    
    Warning: In this setup bigger_side MUST be 640, but it seems it is only enforced
    by hand-picked config and nothing else
    
    Any other bigger_side, and the code will fail, though bigger_side does not matter much when not using TensorRT
    
    """

    def __init__(self, config_dir: str, model_dir: str, imsize, kpts_model=False,
                 crop_fov=False, use_trt=False, device: str = "cuda"):
        if crop_fov:
            self.config = OmegaConf.load(os.path.join(config_dir, "people_detector_crop.yaml"))
        else:
            self.config = OmegaConf.load(os.path.join(config_dir, "people_detector.yaml"))

        if use_trt:
            if kpts_model:
                self.models = {
                    (640,1): YOLO(os.path.join(model_dir, f'{self.config["model_name"]}-pose-640-640-fp16-bs1.engine')),
                    (640,4): YOLO(os.path.join(model_dir, f'{self.config["model_name"]}-pose-640-640-fp16-bs4.engine')),
                }
            else:
                self.models = {
                    (640,1): YOLO(os.path.join(model_dir, f'{self.config["model_name"]}-640-640-fp16-bs1.engine')),
                    (640,4): YOLO(os.path.join(model_dir, f'{self.config["model_name"]}-640-640-fp16-bs4.engine')),
                }
        else:
            model = YOLO(os.path.join(model_dir, f'{self.config["model_name"]}.pt'))

            self.models = {(640,1): model, (640,4): model}

        self.kpts_model = kpts_model
        self.H, self.W = imsize
        self.device = device

    def predict_level(self, image, level):
        assert image.shape[:2] == (self.H, self.W)

        global_top = int(level.crop_global.t * self.H)
        global_bot = int(level.crop_global.b * self.H)
        global_left = int(level.crop_global.l * self.W)
        global_right = int(level.crop_global.r * self.W)
        orig_image = image
        image = image[global_top:global_bot, global_left:global_right]

        H, W = image.shape[:2]
        crop_x = int(level.crop_x * self.W)
        crop_y = int(level.crop_y * self.H)
        step_x = int(crop_x * (1.0 - level.overlap_x))
        step_y = int(crop_y * (1.0 - level.overlap_y))

        preds_lst = []
        batch = []
        x_range = (W+step_x-1) // step_x
        y_range = (H+step_y-1) // step_y
        for i in range(x_range):
            for j in range(y_range):
                offset_x = i * step_x
                offset_y = j * step_y

                crop_x_scaled, crop_y_scaled = int(crop_x * level.scale), int(crop_y * level.scale)
                crop = image[offset_y:offset_y+crop_y, offset_x:offset_x+crop_x]
                orig_crop_shape = crop.shape
                crop = cv2.resize(crop, None, fx=level.scale, fy=level.scale)
                crop_padded = np.zeros((crop_y_scaled, crop_x_scaled, 3), dtype=np.uint8)
                crop_padded[:crop.shape[0], :crop.shape[1]] = crop

                h, w = crop_y_scaled, crop_x_scaled
                hh = h - h%32 + 32
                ww = w - w%32 + 32
                bigger_side = max(hh,ww)

                # crop_padded_padded = np.zeros((hh, ww, 3), dtype=np.uint8)
                # crop_padded_padded[:h,:w] = crop_padded
                # imgsz = (bigger_side, bigger_side)

                crop_padded_padded = np.zeros((bigger_side, bigger_side, 3), dtype=np.uint8)
                crop_padded_padded[:h,:w] = crop_padded
                imgsz = (bigger_side, bigger_side)

                batch.append((crop_padded_padded, (offset_x+global_left, offset_y+global_top)))

                if len(batch) == level.batch_size or (i == x_range-1 and j == y_range-1):
                    model = self.models[(bigger_side, level.batch_size)]
                    preds = model.predict([item[0] for item in batch], imgsz=imgsz, conf=level.conf, device=self.device, classes=[0], iou=level.nms_local, verbose=False)

                    for idx, pred in enumerate(preds):

                        pred.orig_shape = orig_crop_shape

                        clip_min_boxes = torch.Tensor([0, 0, 0, 0]).reshape((1,4)).to(self.device)
                        clip_max_boxes = torch.Tensor([crop.shape[1],crop.shape[0],crop.shape[1],crop.shape[0]]).reshape((1,4)).to(self.device)
                        boxes = pred.boxes.data.clone()
                        boxes[:,:4] = torch.clip(boxes[:,:4], clip_min_boxes, clip_max_boxes) / level.scale
                        pred.update(boxes=boxes)

                        if self.kpts_model:
                            clip_min_keypoints = torch.Tensor([0, 0]).reshape((1,1,2)).to(self.device)
                            clip_max_keypoints = torch.Tensor([crop.shape[1],crop.shape[0]]).reshape((1,1,2)).to(self.device)
                            keypoints = pred.keypoints.data.clone()
                            keypoints[:,:,:2] = torch.clip(keypoints[:,:,:2], clip_min_keypoints, clip_max_keypoints) / level.scale
                            pred.keypoints = Keypoints(keypoints, pred.orig_shape)

                        boxes = pred.boxes.data
                        if boxes.shape[0] > 0:
                            idxs = (((boxes[:,2] - boxes[:,0]) > 1) * ((boxes[:,3] - boxes[:,1]) > 1)) > 0
                            pred = pred[idxs]

                        preds_lst.append((pred, batch[idx][1]))
                    batch = []

        boxes_global = []
        if self.kpts_model:
            keypoints_global = []

        for pred, offset in preds_lst:
            if pred.boxes.shape[0] > 0:
                boxes = pred.boxes.data
                boxes[:,0] += offset[0]
                boxes[:,2] += offset[0]
                boxes[:,1] += offset[1]
                boxes[:,3] += offset[1]
                boxes_global.append(boxes)

                if self.kpts_model:
                    keypoints = pred.keypoints.data
                    keypoints[:,:,0] += offset[0]
                    keypoints[:,:,1] += offset[1]
                    keypoints_global.append(keypoints)
        
        if len(boxes_global) == 0:
            return preds_lst[-1][0]

        boxes_global = torch.cat(boxes_global, dim=0)
        if self.kpts_model:
            keypoints_global = torch.cat(keypoints_global, dim=0)

        if len(preds_lst) > 1:
            idxs = torchvision.ops.nms(boxes_global[:,:4], boxes_global[:,4], level.nms_global)
            boxes_global = boxes_global[idxs]
            if self.kpts_model:
                keypoints_global = keypoints_global[idxs]

        results = Results(
            orig_img=orig_image,
            path=preds_lst[-1][0].path,
            names=preds_lst[-1][0].names,
            boxes=boxes_global,
            keypoints=keypoints_global if self.kpts_model else None,
        )

        return results

    def predict(self, image):
        result_lst = []
        boxes_all = []
        scores_all_nms = []
        if self.kpts_model:
            keypoints_all = []

        for level in self.config.levels:
            result = self.predict_level(image, level.params)
            result_lst.append(result)
            if result.boxes.shape[0] > 0:
                boxes_all.append(result.boxes.data)
                scores_all_nms.append(result.boxes.data[:,4] / level.params.scale**4)
                if self.kpts_model:
                    keypoints_all.append(result.keypoints.data)

        if len(boxes_all) == 0:
            results = result_lst[0]
            results.orig_img = image
            return list()

        boxes_all_tensor = torch.cat(boxes_all, dim=0)
        scores_all_nms_tensor = torch.cat(scores_all_nms, dim=0)

        if len(result_lst) > 1:
            idxs = torchvision.ops.nms(boxes_all_tensor[:, :4], scores_all_nms_tensor, self.config.nms)
            boxes_all_filtered = [boxes_all_tensor[idx] for idx in idxs]

        results = []
        for box in boxes_all_filtered:
            box_np = box.detach().cpu().numpy()
            results.append(DetectedObject(mask=None, bbox=box_np[:4],
                                          name="person", conf=box_np[-2].item(),
                                          imsize=image.shape[:2]))

        return results
