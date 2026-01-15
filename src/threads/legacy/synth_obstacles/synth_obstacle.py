# By Oleksiy Grechnyev, 4/24/24


import sys
import pathlib

import numpy as np
import cv2 as cv


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
class SynthObstacle:
    def __init__(self, obstacle_path):
        p_obst = pathlib.Path(obstacle_path)
        self.mask = cv.imread(str(p_obst / 'mask.png'), cv.IMREAD_GRAYSCALE)
        self.img_i = cv.imread(str(p_obst / 'img_i.png'))
        self.img_nz = cv.imread(str(p_obst / 'img_nz.png'))
        self.img_nxy = cv.imread(str(p_obst / 'img_nxy.png'))
        self.img_nxyz = cv.imread(str(p_obst / 'img_nxyz.png'))
        self.mask_h, self.mask_w = self.mask.shape
        
    def get_size(self):
        return self.mask_w, self.mask_h
    
    @staticmethod
    def calc_position_and_overlap(p, frame_s, mask_s):
        pf1, pf2 = 0, frame_s
        pm1, pm2 = p, p + mask_s
        p1, p2 = max(pf1, pm1), min(pf2, pm2)
        return p1, p1 - p, p2 - p1

    def add_obstacle(self, frame, frame_x, frame_y):
        # img_i, img_nz, img_xy, img_xyz = frame
        
        # Calculate the proper overlap
        frame_h, frame_w = frame[0].shape[:2]
        
        frame_x, mask_x, crop_w = self.calc_position_and_overlap(frame_x, frame_w, self.mask_w)
        frame_y, mask_y, crop_h = self.calc_position_and_overlap(frame_y, frame_h, self.mask_h)

        # print('CROP : ', crop_w, crop_h)
        
        if crop_w <= 0 or crop_h <= 0:
            return # Nothing to do
        
        # Add obstacle image to all 4 channels
        for img_f, img_o in zip(frame, [self.img_i, self.img_nz, self.img_nxy, self.img_nxyz]):
            crop_f = img_f[frame_y: frame_y+crop_h, frame_x: frame_x+crop_w, :]
            crop_o = img_o[mask_y: mask_y+crop_h, mask_x: mask_x+crop_w, :]
            crop_m = self.mask[mask_y: mask_y+crop_h, mask_x: mask_x+crop_w]
            
            cv.copyTo(crop_o, crop_m, crop_f)
        
        
    
    
    
########################################################################################################################
