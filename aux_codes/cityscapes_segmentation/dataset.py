import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from collections import namedtuple
from datasets import load_dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


Label = namedtuple('Label' , [
    'name', 'id', 'trainId', 'category', 'categoryId',
    'hasInstances', 'ignoreInEval', 'color',
    ])

CLASSES = [
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

IDX_TO_NAME = {label.id: label for label in CLASSES}
NAME_TO_IDX = {label.name: label.id for label in CLASSES}


class CityscapesDataset(Dataset):
    def __init__(self, data_split: str = "validation",
                 image_size: List[int] = [640, 1280],
                 augment: bool = True, grayscale: bool = True):
        ds = load_dataset("Chris1/cityscapes_segmentation")
        self.ds = ds[data_split]
        self._image_size = image_size
        self.augment = augment
        self.data_split = data_split
        self.grayscale = int(grayscale)

        # Define transformations
        if self.augment and data_split == "train":
            self.transform = A.Compose([
                # Spatial augmentations
                A.RandomCrop(
                    height=image_size[0],
                    width=image_size[1]
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussianBlur(p=0.2),
                A.ToGray(p=self.grayscale),
                # Min-max scaling to 0-1 range instead of mean/std normalization
                A.ToFloat(max_value=255.0),  # Scales from 0-255 to 0-1
                ToTensorV2(),
            ])
        else:
            # For validation/test, just resize while preserving aspect ratio
            self.transform = A.Compose([
                A.LongestMaxSize(max(image_size)),
                A.RandomCrop(
                    height=image_size[0],
                    width=image_size[1]
                ),
                A.ToGray(p=self.grayscale),
                # Min-max scaling to 0-1 range instead of mean/std normalization
                A.ToFloat(max_value=255.0),  # Scales from 0-255 to 0-1
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get sample from the HuggingFace dataset
        sample = self.ds[idx]

        # Extract image and label
        image = np.array(sample['image'])
        label = np.array(sample['semantic_segmentation'].convert("L"))
        # return image, label

        # Convert segmentation mask to use trainIds
        # This transforms the original IDs to the training IDs specified in CLASSES
        label_trainid = np.zeros_like(label, dtype=np.uint8)
        for cls in CLASSES:
            if cls.id >= 0:  # Skip 'license plate' with id=-1
                mask = (label == cls.id)
                label_trainid[mask] = cls.trainId

        # Create a custom transform for this specific sample that will apply
        # the same transformations to both the image and mask
        if self.data_split == "train" and self.augment:
            # Apply the same transformations to both image and mask
            transformed = self.transform(image=image, mask=label_trainid)
            transformed_image = transformed['image']  # Already a tensor from ToTensorV2
            transformed_mask = transformed['mask']  # Still a numpy array

        else:
            # For validation/test
            transformed = self.transform(image=image, mask=label_trainid)
            transformed_image = transformed['image']  # Already a tensor from ToTensorV2
            transformed_mask = transformed['mask']  # Still a numpy array

        return transformed_image, transformed_mask
