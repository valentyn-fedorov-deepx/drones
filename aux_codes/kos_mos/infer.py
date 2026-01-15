# By Oleksiy Grechnyev, 5/3/24
# Inference of a trained model on our data

import sys
import pathlib

import numpy as np
import cv2 as cv

import torch
import ultralytics

MODEL = './output/trained/fun1.pt'

IMAGE_DIR = '/home/seymour/w/1/vyzai_sniper/stuff/data_seg/splits/fun1/train/images/'


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
def main1():
    model = ultralytics.YOLO(MODEL)
    p_img_dir = pathlib.Path(IMAGE_DIR)

    for p in sorted(p for p in p_img_dir.iterdir()):
        img = cv.imread(str(p))
        model.predict(img, show=True)
        cv.waitKey(0)


########################################################################################################################
if __name__ == '__main__':
    main1()
