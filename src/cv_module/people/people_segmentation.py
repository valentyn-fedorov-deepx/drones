from ultralytics import YOLO
import torch
import cv2
import os
import numpy as np
from typing import List

from src.cv_module.basic_object import DetectedObject


class PeopleSegmenter:
    """
    Find person mask for a person detection BBOX
    runs the YOLOv8-seg model on the padded bbox crop
    """

    def __init__(self, config_dir, model_dir, bigger_side=640,
                 use_trt=False, device: str = "cuda"):
        if use_trt:
            self.seg_model = YOLO(os.path.join(model_dir, f'yolov8m-seg-{bigger_side}-{bigger_side}-fp16-bs1.engine'))
        else:
            self.seg_model = YOLO(os.path.join(model_dir, f'yolov8m-seg.pt'))

        self.crop_pad_x = 1.0
        self.crop_pad_y = 0.5
        self.conf_thr = 0.05

        self.seg_bbox_min_intersection_x = 0.5
        self.seg_bbox_min_intersection_y = 0.25
        self.seg_mask_min_intersection_area = 0.5

        self.target_bigger_side = bigger_side
        self.device = device

    @staticmethod
    def pad_bbox_relative(bbox, pad_x, pad_y, orig_shape):
        x1, y1, x2, y2 = bbox
        dx, dy = x2 - x1, y2 - y1
        xc1, yc1 = max(int(x1 - pad_x*dx), 0), max(int(y1 - pad_y*dy), 0)
        xc2, yc2 = min(int(x2 + pad_x*dx), orig_shape[1]-1), min(int(y2 + pad_y*dy), orig_shape[0]-1)

        return xc1, yc1, xc2, yc2

    @staticmethod
    def get_bbox_intersection(bbox_1, bbox_2):
        x11, y11, x21, y21 = bbox_1
        x12, y12, x22, y22 = bbox_2

        x1, y1, x2, y2 = max(x11, x12), max(y11, y12), min(x21, x22), min(y21, y22)
        if x1 > x2 or y1 > y2:
            return False, None

        return True, (x1, y1, x2, y2)

    def adjust_sizes(self, h, w):
        bigger_side = max(h, w)
        bigger_side = bigger_side - bigger_side % 32 + 32
        bigger_side = min(bigger_side, self.target_bigger_side)
        bigger_side = max(bigger_side, self.target_bigger_side//2)

        if h > w:
            hh = bigger_side
            ww = int(w * bigger_side / h)
            if ww % 32 != 0:
                ww = ww - ww % 32 + 32
        else:
            ww = bigger_side
            hh = int(h * bigger_side / w)
            if hh % 32 != 0:
                hh = hh - hh % 32 + 32

        return (hh, ww), bigger_side

    def process(self, img: np.ndarray, people: List[DetectedObject]):
        H, W = img.shape[:2]

        for person in people:
            bbox_orig = person.bbox.astype(int)
            h_orig, w_orig = bbox_orig[3] - bbox_orig[1], bbox_orig[2] - bbox_orig[0]

            bbox_crop = self.pad_bbox_relative(bbox_orig, self.crop_pad_x, self.crop_pad_y, (H, W))
            h_crop, w_crop = bbox_crop[3] - bbox_crop[1], bbox_crop[2] - bbox_crop[0]

            person_mask = torch.zeros((H, W))
            crop = img[bbox_crop[1]:bbox_crop[3]+1, bbox_crop[0]:bbox_crop[2]+1]
            (h_yolo, w_yolo), bigger_side = self.adjust_sizes(h_crop, w_crop)
            crop_yolo = cv2.resize(crop, (w_yolo, h_yolo))

            pad_x, pad_y = (self.target_bigger_side-w_yolo)//2, (self.target_bigger_side-h_yolo)//2
            crop_yolo_padded = np.zeros((self.target_bigger_side, self.target_bigger_side, 3), dtype=np.uint8)
            crop_yolo_padded[pad_y:h_yolo+pad_y,pad_x:w_yolo+pad_x] = crop_yolo
            crop_yolo = crop_yolo_padded
            bigger_side = self.target_bigger_side

            output = self.seg_model.predict(crop_yolo, conf=self.conf_thr, classes=[0],
                                            imgsz=(bigger_side, bigger_side), verbose=False,
                                            device=self.device)[0]

            for det_idx in range(output.boxes.shape[0]):
                bbox_seg = output.boxes.xyxy[det_idx]
                bbox_seg = bbox_seg[0]-pad_x, bbox_seg[1]-pad_y, bbox_seg[2]-pad_x, bbox_seg[3]-pad_y
                bbox_seg = int(bbox_seg[0]*w_crop/w_yolo), int(bbox_seg[1]*h_crop/h_yolo), int(bbox_seg[2]*w_crop/w_yolo), int(bbox_seg[3]*h_crop/h_yolo)
                bbox_seg = bbox_seg[0]+bbox_crop[0], bbox_seg[1]+bbox_crop[1], bbox_seg[2]+bbox_crop[0], bbox_seg[3]+bbox_crop[1]

                if_intersect, bbox_inter = self.get_bbox_intersection(bbox_seg, bbox_orig)
                if not if_intersect:
                    continue
                h_inter, w_inter = bbox_inter[3] - bbox_inter[1], bbox_inter[2] - bbox_inter[0]

                if (w_inter/w_orig > self.seg_bbox_min_intersection_x) and (h_inter/h_orig > self.seg_bbox_min_intersection_y):
                    mask_cropped = output.masks.data[det_idx]
                    mask_cropped = mask_cropped[pad_y:pad_y+h_yolo,pad_x:pad_x+w_yolo]
                    mask_cropped = mask_cropped.view(1, 1, *mask_cropped.shape)
                    mask_cropped = torch.nn.functional.interpolate(mask_cropped, size=(h_crop, w_crop), mode="bilinear")
                    mask_cropped = mask_cropped[0,0] > 0

                    area_total = mask_cropped.sum()
                    area_inside_bbox = mask_cropped[
                        bbox_inter[1]-bbox_crop[1]:bbox_inter[3]-bbox_crop[1]+1,
                        bbox_inter[0]-bbox_crop[0]:bbox_inter[2]-bbox_crop[0]+1,
                    ].sum()
                    if area_inside_bbox / area_total > self.seg_mask_min_intersection_area:
                        person_mask[bbox_crop[1]:bbox_crop[3],bbox_crop[0]:bbox_crop[2]] = mask_cropped
                        break

            person.mask = person_mask.detach().cpu().numpy()

        return people
