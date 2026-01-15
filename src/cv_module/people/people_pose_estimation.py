from ultralytics import YOLO
import torch
import cv2
import numpy as np
import os
from omegaconf import OmegaConf
from typing import List

from src.cv_module.detected_object import DetectedObject


class PeoplePoseEstimator:
    """
    Find human pose with YOLOv8-pose

    Runs the model on the cropped human bboxes, requires bboxes from detection
    """

    def __init__(self, config_dir, model_dir, bigger_side=640,
                 use_trt=False, device="cuda"):

        self.config = OmegaConf.load(os.path.join(config_dir, "pose_estimator.yaml"))

        if use_trt:
            self.pose_model = YOLO(os.path.join(model_dir, f'{self.config.model_name}-{bigger_side}-{bigger_side}-fp16-bs1.engine'))
        else:
            self.pose_model = YOLO(os.path.join(model_dir, f'{self.config.model_name}.pt'))
        self.target_bigger_side = bigger_side
        self.device = device

    @staticmethod
    def pad_bbox_relative(bbox, pad_x, pad_y, orig_shape):
        x1, y1, x2, y2 = bbox
        dx, dy = x2 - x1, y2 - y1
        xc1, yc1 = max(int(x1 - pad_x*dx), 0), max(int(y1 - pad_y*dy), 0)
        xc2, yc2 = min(int(x2 + pad_x*dx), orig_shape[1]-1), min(int(y2 + pad_y*dy), orig_shape[0]-1)

        return xc1, yc1, xc2, yc2

    def adjust_sizes(self, h, w):
        bigger_side = max(h, w)
        bigger_side = bigger_side - bigger_side % 32 + 32
        bigger_side = min(bigger_side, self.target_bigger_side)
        bigger_side = max(bigger_side, self.target_bigger_side//2)

        if h > w:
            hh = bigger_side
            ww = max(int(w * bigger_side / h), 1)
            if ww % 32 != 0:
                ww = ww - ww % 32 + 32
        else:
            ww = bigger_side
            hh = max(int(h * bigger_side / w), 1)
            if hh % 32 != 0:
                hh = hh - hh % 32 + 32

        return (hh, ww), bigger_side

    @staticmethod
    def get_bbox_intersection(bbox_1, bbox_2):
        x11, y11, x21, y21 = bbox_1
        x12, y12, x22, y22 = bbox_2

        x1, y1, x2, y2 = max(x11, x12), max(y11, y12), min(x21, x22), min(y21, y22)
        if x1 > x2 or y1 > y2:
            return False, None

        return True, (x1, y1, x2, y2)

    def process(self, img: np.ndarray, people: List[DetectedObject]):
        H, W = img.shape[:2]

        kpts_new = []
        idxs_new = []

        for person in people:
            if len(idxs_new) == 5:
                break
            bbox_orig = person.bbox.astype(int)
            h_orig, w_orig = bbox_orig[3] - bbox_orig[1], bbox_orig[2] - bbox_orig[0]

            bbox_crop = self.pad_bbox_relative(bbox_orig, self.config.crop_pad_x, self.config.crop_pad_y, (H, W))
            h_crop, w_crop = bbox_crop[3] - bbox_crop[1], bbox_crop[2] - bbox_crop[0]

            crop = img[bbox_crop[1]:bbox_crop[3]+1,bbox_crop[0]:bbox_crop[2]+1]
            (h_yolo, w_yolo), bigger_side = self.adjust_sizes(h_crop, w_crop)
            crop_yolo = cv2.resize(crop, (w_yolo, h_yolo))

            # pad_x, pad_y = (self.target_bigger_side-w_yolo)//2, (self.target_bigger_side-h_yolo)//2
            # crop_yolo_padded = np.zeros((self.target_bigger_side, self.target_bigger_side, 3), dtype=np.uint8)
            # crop_yolo_padded[pad_y:h_yolo+pad_y,pad_x:w_yolo+pad_x] = crop_yolo
            # crop_yolo = crop_yolo_padded
            bigger_side = self.target_bigger_side

            output = self.pose_model.predict(crop_yolo, conf=self.config.conf_thr, classes=[0],
                                             imgsz=(bigger_side, bigger_side), verbose=False,
                                             device=self.device)[0]

            for det_idx in range(output.boxes.shape[0]):
                bbox_pose = output.boxes.xyxy[det_idx]

                # bbox_pose = bbox_pose[0]-pad_x, bbox_pose[1]-pad_y, bbox_pose[2]-pad_x, bbox_pose[3]-pad_y
                if bbox_pose[0] < 0 or bbox_pose[1] < 0 or bbox_pose[2] > w_yolo or bbox_pose[3] > h_yolo:
                    continue
                # bbox_pose = max(bbox_pose[0], 0), max(bbox_pose[1], 1), min(bbox_pose[2], w_yolo), min(bbox_pose[3], h_yolo)

                bbox_pose = int(bbox_pose[0]*w_crop/w_yolo), int(bbox_pose[1]*h_crop/h_yolo), int(bbox_pose[2]*w_crop/w_yolo), int(bbox_pose[3]*h_crop/h_yolo)
                bbox_pose = bbox_pose[0]+bbox_crop[0], bbox_pose[1]+bbox_crop[1], bbox_pose[2]+bbox_crop[0], bbox_pose[3]+bbox_crop[1]

                if_intersect, bbox_inter = self.get_bbox_intersection(bbox_pose, bbox_orig)
                if not if_intersect:
                    continue
                h_inter, w_inter = bbox_inter[3] - bbox_inter[1], bbox_inter[2] - bbox_inter[0]

                if (w_inter/w_orig > self.config.pose_bbox_min_intersection_x) and (h_inter/h_orig > self.config.pose_bbox_min_intersection_y):
                    keypoints = torch.clone(output.keypoints.data[det_idx:det_idx+1])
                    # keypoints[:,:,0] -= pad_x
                    # keypoints[:,:,1] -= pad_y
                    keypoints[:,:,0] *= w_crop/w_yolo
                    keypoints[:,:,1] *= h_crop/h_yolo
                    keypoints[:,:,0] += bbox_crop[0]
                    keypoints[:,:,1] += bbox_crop[1]

                    inside_box = (keypoints[:,:,0] > bbox_orig[0]) * (keypoints[:,:,0] < bbox_orig[2]) * (keypoints[:,:,1] > bbox_orig[1]) * (keypoints[:,:,1] < bbox_orig[3])
                    if inside_box.sum() >= self.config.min_keypoints_inside_original_bbox:
                        keypoints[~inside_box,:] = 0
                        keypoints = keypoints.detach().cpu().numpy()[0]
                        pose = keypoints[:, :2]
                        pose_conf = keypoints[:, 2]

                        person.keypoints = pose
                        person.keypoints_conf = pose_conf
                        break

        return people
