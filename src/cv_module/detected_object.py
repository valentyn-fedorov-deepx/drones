from typing import Optional, Tuple
from datetime import datetime

import numpy as np


class DetectedObject:
    """
    An obstacle (static or dynamic) which can occlude a human
    Contains a mask and bbox
    TODO: mask2bbox can be optimized
    """
    def __init__(self, mask: Optional[np.ndarray], name: str, conf: float,
                 imsize: Tuple[int], bbox: Optional[Tuple[int]] = None,
                 timestamp: float = None):
        if timestamp is None:
            self.timestamp = datetime.now().timestamp()
        else:
            self.timestamp = timestamp

        if mask is None and bbox is None:
            raise ValueError("Either bbox or mask should be provided")

        self.H, self.W = imsize

        if bbox is None and mask is not None:
            x1, y1, x2, y2 = self.mask2bbox(mask)
        else:
            x1, y1, x2, y2 = bbox

        self.mask = mask

        self.name = name
        self.bbox = np.array((x1, y1, x2, y2)).round().astype(int)
        self.conf = conf
        if mask is not None:
            self.area = self.mask.sum()
        else:
            self.area = (x2 - x1) * (y2 - y1)

        self.dist_abc = np.inf
        self.dist_tmp_history = {}
        self.dist_corrected = np.inf
        self.dist = np.nan

        self.contour = None
        self.mask_visual = None

        self.keypoints = None
        self.keypoints_conf = None

    @staticmethod
    def mask2bbox(mask):
        a = np.where(mask > 0)
        bbox = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])
        return bbox

    def get_full_mask(self):
        full_mask = np.zeros((self.H, self.W))
        full_mask[self.bbox[1]:self.bbox[3]+1,self.bbox[0]:self.bbox[2]+1] = self.mask
        return full_mask

    def get_tracking_points(self):
        x1, y1, x2, y2 = self.bbox

        return np.array([(x1, y1), (x2, y2)])

    def get_tracking_points_conf(self):
        return np.full(self.get_tracking_points().shape[0],
                       fill_value=self.conf)

    @staticmethod
    def get_bbox_intersection(bbox_1, bbox_2):
        x11, y11, x21, y21 = bbox_1
        x12, y12, x22, y22 = bbox_2

        x1, y1, x2, y2 = max(x11, x12), max(y11, y12), min(x21, x22), min(y21, y22)
        if x1 > x2 or y1 > y2:
            return False, None

        return True, (x1, y1, x2, y2)

    @staticmethod
    def get_mask_intersection(mask_1, bbox_1, mask_2, bbox_2):
        if_intersect, bbox = DetectedObject.get_bbox_intersection(bbox_1, bbox_2)

        if not if_intersect:
            return 0

        x1, y1, x2, y2 = bbox
        x11, y11 = bbox_1[:2]
        x12, y12 = bbox_2[:2]

        mask_intersection_1 = mask_1[y1-y11:y2-y11+1, x1-x11:x2-x11+1]
        mask_intersection_2 = mask_2[y1-y12:y2-y12+1, x1-x12:x2-x12+1]
        mask_intersection = mask_intersection_1 * mask_intersection_2

        return mask_intersection.sum()

    def intersection_area(self, mask, bbox=None):

        mask_1, bbox_1 = self.mask, self.bbox
        if bbox is None:
            mask_2, bbox_2 = mask, self.mask2bbox(mask)
        else:
            mask_2, bbox_2 = mask, bbox

        return self.get_mask_intersection(mask_1, bbox_1, mask_2, bbox_2)

    def is_inside(self, x, y):
        if x < self.bbox[0] or y < self.bbox[1] or x > self.bbox[2] or y > self.bbox[3]:
            return False
        return self.mask[y-self.bbox[1],x-self.bbox[0]]

    def __repr__(self):
        return f"{self.name.upper()}, conf: {self.conf}"

    def __str__(self):
        return f"{self.name.upper()}, conf: {self.conf}"
