import sys

import torch
from ultralytics import YOLO
import os
import cv2

from src.cv_module.obstacles.obstacle import Obstacle
from src.cv_module.cars.car import CarDetection



########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
class DynamicObstacleDetector:
    """
    Detect dynamic obstacles (cars OR obstacles) with a YOLOv8 segmentation model
    """
    def __init__(self, config_dir, model_dir, detect_mode, imsize, imgsz_infer,
                 use_trt=False, device: str = "cuda"):
        self.H, self.W = imsize
        self.imgsz = imgsz_infer
        self.detect_mode = detect_mode

        if detect_mode == 'cars':
            prefix = ''
            self.classes = [2, 5, 7]
            self.class_names = {2: 'Car', 5: 'Bus', 7: 'Truck'}
            self.create_object = CarDetection
        elif detect_mode == 'obstacles':
            prefix = 'obst-'
            self.classes = [0, 1, 2, 3]
            # self.class_names = ['hedge', 'obstacle', 'post', 'tree trunk']
            # self.class_names = ['Tree', 'obstacle', 'bush', 'pole']
            self.class_names = ['tree trunk', 'obstacle', 'hedge', 'post']
            self.create_object = Obstacle

        if use_trt:
            self.model = YOLO(
                os.path.join(model_dir, f'{prefix}yolov8m-seg-{imgsz_infer[0]}-{imgsz_infer[1]}-fp16-bs1.engine'))
        else:
            self.model = YOLO(os.path.join(model_dir, f'{prefix}yolov8m-seg.pt'))

        self.conf_thr = 0.4
        self.nms_iou = 0.5
        self.device = device

    def process(self, image):
        image = cv2.resize(image, self.imgsz)

        results = self.model.predict(image, imgsz=self.imgsz[::-1], conf=self.conf_thr, device=self.device,
                                     classes=self.classes, iou=self.nms_iou, verbose=False)[0]

        obstacles = []
        if results.masks is not None:
            for idx in range(len(results.boxes)):
                mask = results.masks[idx].data[0]
                bbox = results.boxes[idx].xyxy[0].cpu().numpy().astype(int)
                cls = int(results.boxes[idx].cls[0].cpu().numpy())
                # print('imgsz=', self.imgsz)
                # print('bbox=', bbox)
                # print_it(mask, 'mask')

                if mask[bbox[1]:bbox[3], bbox[0]:bbox[2]].sum() == 0:
                    continue
                bbox = int(bbox[0] * self.W / self.imgsz[0]), int(bbox[1] * self.H / self.imgsz[1]), int(
                    bbox[2] * self.W / self.imgsz[0]), int(bbox[3] * self.H / self.imgsz[1])
                mask = torch.nn.functional.interpolate(mask[None, None], size=(self.H, self.W), mode="bilinear")[
                           0, 0] > 0
                mask = mask.cpu().numpy()
                conf = results.boxes[idx].data[0][-2]
                phrase = self.class_names[cls]
                obstacles.append(self.create_object(mask=mask, phrase=phrase,
                                                    conf=conf, imsize=(self.H, self.W),
                                                    bbox=bbox, bbox_recompute=True))

        return obstacles

#     @staticmethod
#     def show_mask(mask, image, random_color=True):
#         import numpy as np
#         from PIL import Image
#         if random_color:
#             color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
#         else:
#             color = np.array([30/255, 144/255, 255/255, 0.6])
#         h, w = mask.shape[-2:]
#         mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#
#         annotated_frame_pil = Image.fromarray(image).convert("RGBA")
#         mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")
#
#         return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))
#
