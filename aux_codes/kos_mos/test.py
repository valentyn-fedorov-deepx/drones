# By Oleksiy Grechnyev, 5/3/24
# Run a trained model on a category and compare with the ground truthe (from DINO+SAM)

import sys
import pathlib

import numpy as np
import cv2 as cv

import torch
import ultralytics

MODEL = './output/trained/fun1.pt'
ROOT_DATA_COLLECTION = '/home/seymour/w/1/vyzai_sniper/stuff/data_seg/data'
CATEGORY = 'test'


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
def plot_semseg_mask(img, semseg_mask):
    """Highlight a 3-channel grayscale image with seg mask colors"""
    img = img.astype('int16')

    colors = [
            (0, 0, 0),  # Black BG
            (0, 0xff, 0),  # Green hedge
            (0, 0, 0xff),  # Red obstacle
            (0xff, 0, 0),  # Blue POST
            (0, 0xff, 0xff),  # Yellow tree trunk
        ]

    for ic in range(1, 5):
        m = (semseg_mask == ic)
        img[m, :] = (2 * img[m, :] + colors[ic]) // 3

    img = img.astype('uint8')
    return img


########################################################################################################################
def predict_semseg_mask(model, img):
    """Run instance segmentation and convert it to semseg_mask"""
    im_h, im_w = img.shape[:2]
    res = model.predict(img)  # boxes, masks, prob, keypoints, obb
    # res = model.predict(img, imgsz=608)  # boxes, masks, prob, keypoints, obb

    # print_it(res[0].masks.data, 'masks')  # [17, 544, 640]
    # print_it(res[0].boxes.data, 'boxes')  # [17, 6]
    # print_it(res[0].boxes.cls.data, 'cls')  # [17]

    masks = res[0].masks.data.detach().cpu().numpy()
    classes = res[0].boxes.cls.data.detach().cpu().numpy().astype('int')

    seg_mask = np.zeros((im_h, im_w), dtype='uint8')
    for mask, cls in zip(masks, classes):
        # Assuming inference size == image size == (640, 512), otherwise resize
        seg_mask[mask > 0] = cls + 1   # For semseg class, add 1 to the instance seg class

    return seg_mask


########################################################################################################################
def main1():
    p_data = pathlib.Path(ROOT_DATA_COLLECTION) / CATEGORY
    p_item_list = sorted([p for p in p_data.iterdir()])

    model = ultralytics.YOLO(MODEL)

    for p_item in p_item_list:
        # Read data
        img = cv.imread(str(p_item / 'img.png'))
        semseg_mask_gt = cv.imread(str(p_item / 'semseg_mask.png'), cv.IMREAD_GRAYSCALE)
        semseg_mask_pred = predict_semseg_mask(model, img)

        img_gt = plot_semseg_mask(img, semseg_mask_gt)
        img_pred = plot_semseg_mask(img, semseg_mask_pred)

        cv.imshow('img_gt', img_gt)
        cv.imshow('img_pred', img_pred)
        if 27 == cv.waitKey(0):
            sys.exit()


########################################################################################################################
if __name__ == '__main__':
    main1()
