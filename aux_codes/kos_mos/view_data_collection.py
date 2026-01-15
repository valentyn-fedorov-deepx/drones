# By Oleksiy Grechnyev, 5/1/24
# View images from data collection (with labels)

import sys
import pathlib

import numpy as np
import cv2 as cv

ROOT_DATA_COLLECTION = '/home/seymour/w/1/vyzai_sniper/stuff/data_seg/data'
CATEGORY = 'test'
DRAW_ALL = True


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
def visualize_all(img, labels, name_in):
    im_h, im_w = img.shape[:2]
    colors = [
        (0, 0xff, 0),  # Green hedge
        (0, 0, 0xff),  # Red obstacle
        (0xff, 0, 0),  # Blue POST
        (0, 0xff, 0xff),  # Yellow tree trunk
    ]
    img_vis = img.copy()
    for label_str in labels:
        label_str = label_str.split()
        cls = int(label_str[0])
        label = np.array([float(x) for x in label_str[1:]])
        contour = label.reshape(-1, 1, 2)
        contour = (contour * [im_w, im_h]).astype('int')
        cv.drawContours(img_vis, [contour], 0, colors[cls], 1)

    cv.imshow('img_vis', img_vis)
    if 27 == cv.waitKey(0):
        sys.exit()


########################################################################################################################
def visualize_interactive(img, labels, name_in):
    im_h, im_w = img.shape[:2]
    colors = [
        (0, 0xff, 0),  # Green hedge
        (0, 0, 0xff),  # Red obstacle
        (0xff, 0, 0),  # Blue POST
        (0, 0xff, 0xff),  # Yellow tree trunk
    ]
    for label_str in labels:
        img_vis = img.copy()
        label_str = label_str.split()
        cls = int(label_str[0])
        label = np.array([float(x) for x in label_str[1:]])
        contour = label.reshape(-1, 1, 2)
        contour = (contour * [im_w, im_h]).astype('int')
        cv.drawContours(img_vis, [contour], 0, colors[cls], 1)
        cv.imshow('img_vis', img_vis)
        if 27 == cv.waitKey(0):
            sys.exit()


########################################################################################################################
def main():
    p_data = pathlib.Path(ROOT_DATA_COLLECTION) / CATEGORY
    p_item_list = sorted([p for p in p_data.iterdir()])

    for p_item in p_item_list:
        # Read data
        img = cv.imread(str(p_item / 'img.png'))

        with open(p_item / 'labels.txt') as f:
            labels = [line.strip() for line in f]

        with open(p_item / 'name_in.txt') as f:
            name_in = f.read()

        if DRAW_ALL:
            visualize_all(img, labels, name_in)
        else:
            visualize_interactive(img, labels, name_in)


########################################################################################################################
if __name__ == '__main__':
    main()
