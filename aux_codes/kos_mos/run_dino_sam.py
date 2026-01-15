# By Oleksiy Grechnyev, 5/1/24
#
# Run input photos via GroundingDINO+SAM (static_obstacle_detection.py) to create instance segmentation labels
# These labels can be used to train YOLOv8-seg
# Also crops and scales images to the 608x512 resolution and converts them to grayscale
# Data management: see README. This code processes   **Raw image collection** into **Data collection**.


import sys
import pathlib
import shutil

import numpy as np
import cv2 as cv

import vol.static_obstacle_detection

ROOT_RAW_IMAGE_COLLECTION = '/home/seymour/w/1/vyzai_sniper/stuff/data_seg/raw_images'
ROOT_DATA_COLLECTION = '/home/seymour/w/1/vyzai_sniper/stuff/data_seg/data'

# CATEGORIES_TO_PROCESS = None  # Process all categories
CATEGORIES_TO_PROCESS = ['test']   # Selected categories only

GROUNDING_DINO_DIR = '/home/seymour/w/1/vyzai_sniper/people_track_python/code_algo/GroundingDINO'
SAM_DIR = '/home/seymour/w/1/vyzai_sniper/people_track_python/code_algo/models'
FORCE_REDO = False  # Recalculate everything, deleting any previous results


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
def downsize(img):
    img = cv.resize(img, None, None, 0.5, 0.5)
    return img


def auto_downsize(img):
    while max(img.shape[0], img.shape[1]) > 1920:
        img = downsize(img)
    return img


########################################################################################################################
class Engine:
    """DINO+SAM engine, performs the actual processing"""

    def __init__(self):
        self.sob = vol.static_obstacle_detection.StaticObstacleDetector(GROUNDING_DINO_DIR, SAM_DIR, (0, 0))
        self.class_names = ['hedge', 'obstacle', 'post', 'tree trunk']
        self.class_names_semantic = ['background', 'hedge', 'obstacle', 'post', 'tree trunk']
        self.inference_size = (608, 512)

    def crop_image(self, img):
        # Crop the image to achieve the required aspect ratio
        im_h0, im_w0 = img.shape[:2]
        desired_aspect = 2448 / 2048
        if im_w0 / im_h0 > desired_aspect:
            # Horizontal crop
            img_w = int(im_h0 * desired_aspect)
            pos_x = (im_w0 - img_w) // 2
            img_crop = img[:, pos_x:pos_x + img_w, :]
        else:
            # Vertical crop
            img_h = int(im_w0 / desired_aspect)
            pos_y = (im_h0 - img_h) // 2
            img_crop = img[pos_y:pos_y + img_h, :, :]

        return img_crop.copy()

    def create_contour_labels(self, im_shape, obstacles, masks):
        """Create contour labels from masks for YOLOv8-seg training (instance seg)"""
        im_h, im_w = im_shape[:2]
        eps = max(im_w, im_h) // 200
        labels = []
        for ob, mask in zip(obstacles, masks):
            cls = self.class_names.index(ob.phrase)

            # Find contour, which we use as labels in YOLOv8-seg
            contours, hier = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1)
            assert len(contours) > 0

            # Select the largest contour and downsample it
            areas = [cv.contourArea(c) for c in contours]
            idx = np.argmax(areas)
            contour0 = contours[idx]
            if len(contour0) <= 30:
                contour = contour0
            else:
                contour = cv.approxPolyDP(contour0, eps, True)
                if len(contour) < 10:
                    contour = cv.approxPolyDP(contour0, eps / 2, True)

            if False:
                # Visualize
                mask_vis = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
                cv.drawContours(mask_vis, [contour], 0, (0xff, 0, 0xff), 3)
                cv.imshow('mask', downsize(mask_vis))
                if 27 == cv.waitKey(0):
                    sys.exit()

            label = contour[:, 0, :].astype('float64') / [im_w, im_h]
            label = label.ravel()
            label = [cls, *label]
            labels.append(label)

        return labels

    def create_semantic_seg_mask(self, im_shape, obstacles, masks):
        """Create mask for semantic segmentation, and its visualization"""
        im_h, im_w = im_shape[:2]
        seg_mask = np.zeros((im_h, im_w), dtype='uint8')
        for ob, mask in zip(obstacles, masks):
            for ic, cls_name in enumerate(self.class_names_semantic):
                if ob.phrase == cls_name:
                    seg_mask[mask > 0] = ic

        # Visualize
        seg_mask_vis = np.zeros((im_h, im_w, 3), dtype='uint8')
        colors = [
            (0, 0, 0),  # Black BG
            (0, 0xff, 0),  # Green hedge
            (0, 0, 0xff),  # Red obstacle
            (0xff, 0, 0),  # Blue POST
            (0, 0xff, 0xff),  # Yellow tree trunk
        ]

        for i in range(seg_mask.max() + 1):
            seg_mask_vis[seg_mask == i, :] = colors[i]

        # Scale to desired resolution
        seg_mask = cv.resize(seg_mask, self.inference_size, interpolation=cv.INTER_NEAREST)
        seg_mask_vis = cv.resize(seg_mask_vis, self.inference_size, interpolation=cv.INTER_NEAREST)

        return seg_mask, seg_mask_vis

    def process(self, img: np.ndarray) -> dict:
        """Process one input image"""
        # Prepare image
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_gray3 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # Grayscale 3 channels

        # Crop with a correct aspect ratio
        img_crop = self.crop_image(img_gray3)

        # Run through the detector
        im_h, im_w = img_crop.shape[:2]
        self.sob.W = im_w
        self.sob.H = im_h
        obstacles = self.sob.process(img_crop)
        masks = [ob.get_full_mask() for ob in obstacles]
        masks = [(m > 0).astype('uint8') * 255 for m in masks]

        # Create contour labels from masks for instance seg
        contour_labels = self.create_contour_labels(img_crop.shape, obstacles, masks)

        # Create mask for semantic seg
        semseg_mask, semseg_mask_vis = self.create_semantic_seg_mask(img_crop.shape, obstacles, masks)

        # Scale the image and put everything together
        img_crop_scaled = cv.resize(img_crop, self.inference_size)
        data = {
            'img': img_crop_scaled,
            'contour_labels': contour_labels,
            'semseg_mask': semseg_mask,
            'semseg_mask_vis': semseg_mask_vis,
        }
        return data


########################################################################################################################
def process_one_image(p_in, p_out, engine):
    """Process one image and write the result"""
    img = cv.imread(str(p_in), cv.IMREAD_GRAYSCALE)
    assert img is not None

    # Run image through DINO+SAM
    data = engine.process(img)

    # Write the results
    cv.imwrite(str(p_out / 'img.png'), data['img'])
    cv.imwrite(str(p_out / 'semseg_mask.png'), data['semseg_mask'])
    cv.imwrite(str(p_out / 'semseg_mask_vis.png'), data['semseg_mask_vis'])
    with open(p_out / 'labels.txt', 'w') as f:
        for label in data['contour_labels']:
            print(' '.join([str(x) for x in label]), file=f)

    # Mark that we finished successfully and write the original image name
    with open(p_out / 'name_in.txt', 'w') as f:
        f.write(p_in.name)


########################################################################################################################
def create_already_processed_list(p_data_cat):
    """Also removes corrupt subdirectories"""
    result = set()
    plist = [p for p in p_data_cat.iterdir()]

    for p in plist:
        if not p.is_dir() or not (p / 'name_in.txt').exists():  # Garbage or corrupt directory
            shutil.rmtree(p)
            continue

        with open(p / 'name_in.txt') as f:
            name = f.read().strip()
        result.add(name)

    return result


########################################################################################################################
def process_one_category(p_raw_cat, p_data_cat, cat, engine):
    # Input image list
    input_image_list = sorted(p.name for p in p_raw_cat.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png'])

    # Images already processed, skip them
    already_processed_image_list = create_already_processed_list(p_data_cat)

    # Run over all images
    idx = 0
    for name_in in input_image_list:
        print(f'Prccessing {name_in}')
        if name_in in already_processed_image_list:
            print(f'{name_in} is already processed, skipping')
            continue

        # Find the next available index
        while (p_data_cat / f'{idx:06d}').exists():
            idx += 1

        p_out = p_data_cat / f'{idx:06d}'
        p_out.mkdir()
        process_one_image(p_raw_cat / name_in, p_out, engine)
        idx += 1


########################################################################################################################
def main():
    p_root_raw = pathlib.Path(ROOT_RAW_IMAGE_COLLECTION)
    p_root_data = pathlib.Path(ROOT_DATA_COLLECTION)

    # Find all categories to process
    categories = []
    for p in p_root_raw.iterdir():
        if p.is_dir():
            cat = p.name
            if CATEGORIES_TO_PROCESS is None or cat in CATEGORIES_TO_PROCESS:
                categories.append(cat)
    categories.sort()
    print('categories=', categories)

    # Create DINO+SAM engine
    engine = Engine()

    # Process each category
    for cat in categories:
        print('=====================================================')
        print('Processing category', cat)
        p_raw_cat = p_root_raw / cat
        p_data_cat = p_root_data / cat
        if FORCE_REDO:
            shutil.rmtree(p_data_cat)
        p_data_cat.mkdir(parents=True, exist_ok=True)
        process_one_category(p_raw_cat, p_data_cat, cat, engine)


########################################################################################################################
if __name__ == '__main__':
    main()
