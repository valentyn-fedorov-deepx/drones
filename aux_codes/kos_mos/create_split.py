# By Oleksiy Grechnyev, 5/1/24
# Create a split from processed data, which is directly usable for YOLOv8 training

import sys
import pathlib
import shutil
import random

# import numpy as np
# import cv2 as cv

ROOT_DATA_COLLECTION = '/home/seymour/w/1/vyzai_sniper/stuff/data_seg/data'
ROOT_SPLITS = '/home/seymour/w/1/vyzai_sniper/stuff/data_seg/splits'
SPLIT_NAME = 'fun1'
SHUFFLE = True

random.seed(2022)

CATEGORIES_TO_PROCESS = None  # Process all categories

SPLIT_FRACTIONS = {
    'train': 0.8,
    'val': -1,      # All remaining
}


# CATEGORIES_TO_PROCESS = ['parking_lot1']   # Selected categories only


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
def create_total_list(p_root_data):
    total_list = []
    for p_cat in sorted(p_root_data.iterdir()):
        if not p_cat.is_dir():
            continue
        cat = p_cat.name
        if cat == 'test':
            continue
        if CATEGORIES_TO_PROCESS is None or cat in CATEGORIES_TO_PROCESS:
            # Work with this category
            for p in sorted(p_cat.iterdir()):
                total_list.append((cat, p))

    if SHUFFLE:
        random.shuffle(total_list)

    return total_list


########################################################################################################################
def break_list(total_list, fractions):
    """To train+val spit"""
    split_list = {}
    pos = 0
    n = len(total_list)
    n_frac = len(fractions)

    for i, (name, f) in enumerate(fractions.items()):
        idx1 = pos

        if f < 0:
            # All remaining data, only allowed for the last one
            assert i == n_frac - 1
            idx2 = n
        else:
            assert pos < n
            idx2 = min(n, pos + int(round(f * n)))
        split_list[name] = total_list[idx1:idx2]
        pos = idx2
    return split_list


########################################################################################################################
def write_subsplit(p_subsplit, sub_list):
    """Write train or val"""
    p_images = p_subsplit / 'images'
    p_labels = p_subsplit / 'labels'
    p_images.mkdir()
    p_labels.mkdir()

    for cat, p_in in sub_list:
        num = p_in.name
        shutil.copyfile(p_in / 'img.png', p_images / f'{cat}_{num}.png')
        shutil.copyfile(p_in / 'labels.txt', p_labels / f'{cat}_{num}.txt')


########################################################################################################################
def main():
    p_root_data = pathlib.Path(ROOT_DATA_COLLECTION)
    p_split = pathlib.Path(ROOT_SPLITS) / SPLIT_NAME
    assert not p_split.exists()

    total_list = create_total_list(p_root_data)

    # Break the total list to train + val sets
    split_list = break_list(total_list, SPLIT_FRACTIONS)

    # Copy to where
    p_split.mkdir(parents=True)
    shutil.copy('./data.yaml', p_split / 'data.yaml')
    for name, sub_list in split_list.items():
        p_subsplit = p_split / name
        p_subsplit.mkdir()
        write_subsplit(p_subsplit, sub_list)


########################################################################################################################
if __name__ == '__main__':
    main()
