from argparse import ArgumentParser
from pathlib import Path
import yaml
import numpy as np
import cv2
from typing import Optional, Dict
import torch
import json
from ultralytics import YOLO
import onnx
import onnxruntime

from .model import PartSegmentationModel
from .data.data import create_model_input_transform, draw_groups

from src.cv_module.detectors import YoloDetector
from src.cv_module.obstacles.dynamic_obstacle_detection import DynamicObstacleDetector
from .export import to_numpy


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--source-path", type=Path, required=True,
                        help="Path to video, image, or folder of images")

    parser.add_argument("--config-path", required=True,
                        help="Path to the model config file")

    parser.add_argument("--save-path", type=Path, default="output",
                        help="Results will be saved here")

    parser.add_argument("--ckpt-path",
                        help="Path to the checkpoint file")

    parser.add_argument("--object", choices=["car", "person"],
                        default="person",
                        help="Specify the object that will be splitted in parts")

    parser.add_argument("--device", default="cuda")

    parser.add_argument('--pos', type=int, help='Starting position in frames')

    parser.add_argument('--duration', type=int, help='Video duration from in frames.')

    parser.add_argument("--add-pose", action="store_true", default=False,
                        help="Person pose vizualization will be added to the separate channel")

    parser.add_argument("--pose-config", type=Path,
                        help="Path to the config that contains information about pose kepypoints")

    parser.add_argument("--opacity", type=float, default=0.8,
                        help="Opacity of the inserted mask")

    return parser.parse_args()


def preprocess_input(image: np.ndarray, image_size: int, estimated_pose=None,
                     pose_config: Dict = None):
    image = image.copy()
    height, width = image.shape[:2]

    if height > width:
        pad_value = height - width
        pad_left = pad_value // 2
        pad_right = pad_value - pad_left
        pad_top = 0
        pad_bottom = 0
    elif height < width:
        pad_value = width - height
        pad_top = pad_value // 2
        pad_bottom = pad_value - pad_top
        pad_left = 0
        pad_right = 0
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    if estimated_pose is not None:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask = draw_groups(mask, estimated_pose, pose_config)
        image[:, :, -1] = mask

    padded = cv2.copyMakeBorder(image, pad_top, pad_bottom,
                                pad_left, pad_right, cv2.BORDER_CONSTANT, None, 0)

    final = cv2.resize(padded, image_size)

    params = dict(pad_top=pad_top, pad_bottom=pad_bottom,
                  pad_left=pad_left, pad_right=pad_right,
                  initial_image_size=image.shape[:2],
                  padded_image_size=padded.shape[:2],
                  final_image_size=final.shape[:2])

    return final, params


def postprocess_output(model_output, params: Optional[Dict] = None,
                       max_mask_value: int = 9):
    output_mask = model_output[0].argmax(0).detach().cpu().numpy().astype(np.uint8)
    output_mask = cv2.resize(output_mask, params["padded_image_size"][::-1],
                             interpolation=cv2.INTER_LINEAR)
    if params is not None:
        output_mask = output_mask[params["pad_top"]:output_mask.shape[0] - params["pad_bottom"],
                                  params["pad_left"]:output_mask.shape[1] - params["pad_right"]]

    output_mask *= 255 // max_mask_value

    output_mask = cv2.resize(output_mask, params["initial_image_size"][::-1],
                             interpolation=cv2.INTER_LINEAR)

    return output_mask


def overlay_part_mask(original_frame: np.ndarray, part_mask: np.ndarray, opacity: float = 0.6):
    if not part_mask.any():
        return original_frame

    new_frame = original_frame.copy()
    mask = np.uint8(part_mask > 0)

    indices = np.where(mask)
    mask_values = np.expand_dims(part_mask[indices], -1)
    old_values = new_frame[indices]
    new_frame[indices] = ((mask_values * opacity) + (old_values * (1 - opacity))) // 2

    return new_frame


class ONNXModelWrapper:
    def __init__(self, weights_path: str):
        self.onnx_model = onnx.load(weights_path)
        onnx.checker.check_model(self.onnx_model)
        self.ort_session = onnxruntime.InferenceSession(weights_path,
                                                        providers=["CPUExecutionProvider"])
        self.device = 'cpu'

    def infer(self, image: np.ndarray):
        pass

    def to(self, device: str):
        pass

    def __call__(self, input_image: torch.tensor):
        np_input = to_numpy(input_image).astype(np.float32)
        ort_inputs = {self.ort_session.get_inputs()[0].name: np_input.astype(np.float32)}
        # import ipdb; ipdb.set_trace()
        ort_outs = self.ort_session.run(None, ort_inputs)

        return torch.tensor(ort_outs)[0]


if __name__ == "__main__":
    args = parse_args()
    print(args)

    args.save_path.mkdir(parents=True, exist_ok=True)

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    transform = create_model_input_transform(config["data"]["image_size"])

    if args.ckpt_path.endswith(".ckpt"):
        model = PartSegmentationModel.load_from_checkpoint(args.ckpt_path, **config["model"],
                                                        n_classes=config["data"]["n_classes"],
                                                        map_location=args.device).eval()
    elif args.ckpt_path.endswith(".onnx"):
        model = ONNXModelWrapper(str(args.ckpt_path))

    config_dir = Path("configs")
    model_dir = Path("models")
    imsize = H, W = 2048, 2448

    pose_config = None
    if args.object == "person":
        detector = YoloDetector(config_dir, model_dir, imsize)
        if args.add_pose:
            pose_estimator = YOLO(model_dir / "yolov8m-pose.pt")
            with open(args.pose_config, "r") as f:
                pose_config = json.load(f)
    else:
        imgsz_infer = (640, 640)
        detector = DynamicObstacleDetector(config_dir, model_dir, "cars",
                                           imsize, imgsz_infer, use_trt=False)

    video = cv2.VideoCapture(str(args.source_path))

    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    new_video = cv2.VideoWriter(str(args.save_path / "video.mp4"),
                                fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                                fps=fps, frameSize=(width, height))

    if args.pos:
        video.set(cv2.CAP_PROP_POS_FRAMES, args.pos-1)

    processed_frames = 0
    while True:
        if args.duration and processed_frames > args.duration:
            break
        res, frame = video.read()

        if frame is None:
            break

        frame_copy_to_draw = frame.copy()

        if args.object == "person":
            output = detector.predict(frame)
            if not output:
                continue
            items = [item.bbox for item in output]
        else:
            output = detector.process(frame)
            items = [obstacle.bbox for obstacle in output]

        if not items:
            continue

        for box in items:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            frame_copy_to_draw = cv2.rectangle(frame_copy_to_draw, (x1, y1), (x2, y2), (255, 255, 0), 4)

            if args.add_pose:
                results = pose_estimator(frame[y1:y2, x1:x2])[0]
                all_instances_info = [torch.cat([data, results.keypoints.conf[None, idx].T], dim=1).tolist() for idx, data in enumerate(results.keypoints.data) if data.numel() > 0]
                if not all_instances_info:
                    all_instances_info = None
                else:
                    all_instances_info = np.array(all_instances_info).reshape(-1, 4)
            else:
                all_instances_info = None

            person_crop, preprocessing_params = preprocess_input(frame[y1:y2, x1:x2],
                                                                 (config["data"]["image_size"], config["data"]["image_size"]),
                                                                 all_instances_info, pose_config)

            model_input = transform(image=person_crop)["image"]
            model_output = model(model_input.unsqueeze(0).to(model.device))
            model_mask_np = postprocess_output(model_output, preprocessing_params,
                                               config["data"]["n_classes"])

            frame_copy_to_draw[y1:y2, x1:x2] = overlay_part_mask(frame[y1:y2, x1:x2], model_mask_np, args.opacity)

        new_video.write(frame_copy_to_draw)
        processed_frames += 1

    video.release()
    new_video.release()
