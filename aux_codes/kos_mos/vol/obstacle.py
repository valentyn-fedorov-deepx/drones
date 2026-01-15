import numpy as np


class Obstacle:

    def __init__(self, mask, phrase, conf, imsize, bbox=None, bbox_recompute=False):

        self.H, self.W = imsize

        if bbox is None:
            x1, y1, x2, y2 = self.mask2bbox(mask)
        else:
            x1, y1, x2, y2 = bbox
        self.mask = mask[y1:y2+1, x1:x2+1] > 0
        if bbox_recompute:
            x1_, y1_, x2_, y2_ = self.mask2bbox(self.mask)
            self.mask = self.mask[y1_:y2_+1, x1_:x2_+1] > 0
            x1, y1, x2, y2 = x1+x1_, y1+y1_, x1+x2_, y1+y2_

        self.phrase = phrase
        self.bbox = (x1, y1, x2, y2)
        self.conf = conf

        self.area = self.mask.sum()

        self.dist_abc = np.inf
        self.dist_tmp_history = {}
        self.dist_corrected = np.inf
        self.dist = np.nan

        self.contour = None
        self.mask_visual = None

    @staticmethod
    def mask2bbox(mask):
        a = np.where(mask > 0)
        bbox = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])
        return bbox

    def get_full_mask(self):
        full_mask = np.zeros((self.H, self.W))
        full_mask[self.bbox[1]:self.bbox[3]+1,self.bbox[0]:self.bbox[2]+1] = self.mask
        return full_mask

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
        if_intersect, bbox = Obstacle.get_bbox_intersection(bbox_1, bbox_2)

        if not if_intersect:
            return 0

        x1, y1, x2, y2 = bbox
        x11, y11 = bbox_1[:2]
        x12, y12 = bbox_2[:2]

        mask_intersection_1 = mask_1[y1-y11:y2-y11+1, x1-x11:x2-x11+1]
        mask_intersection_2 = mask_2[y1-y12:y2-y12+1, x1-x12:x2-x12+1]
        mask_intersection = mask_intersection_1 * mask_intersection_2

        return mask_intersection.sum()

    def update_dist(self, A, a, b, c):
        x, y = (self.bbox[0]+self.bbox[2]) / 2, self.bbox[3]
        dist_to_horizon = a*x + b*y + c
        if dist_to_horizon < 0:
            self.dist_abc = np.nan
        else:
            self.dist_abc = A / dist_to_horizon

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
