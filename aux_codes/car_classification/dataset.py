from torch.utils.data import Dataset
from typing import Union, Dict
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def create_augmentations(imsize: int):
    transforms = A.Compose([
        A.AdvancedBlur(),
        A.ColorJitter(),
        A.ISONoise(),
        A.ImageCompression(),
        A.RandomBrightnessContrast(),
        A.HueSaturationValue(),
        A.GaussNoise(var_limit=(150, 500)),
        A.HorizontalFlip(),
        A.SafeRotate(limit=(-15, 15),
                     border_mode=cv2.BORDER_CONSTANT,
                     value=0),
        A.Perspective(),
        A.CoarseDropout(mask_fill_value=0, max_holes=2,
                        max_height=0.3, max_width=0.3,
                        min_height=0.1, min_width=0.1,
                        p=0.1),
    ])

    return transforms


class StanfordCarClassificationDataset(Dataset):
    def __init__(self, data_path: Union[str, Path], classes_config: Dict,
                 grayscale: bool = True, use_augs: bool = False, imsize: int = 256):
        self._classes_config = classes_config

        self._data_path = Path(data_path)
        self._items_path = self._load_data()
        self._grayscale = grayscale
        self._use_augs = use_augs
        if use_augs:
            self._transforms = create_augmentations(imsize)
        self._to_tensor = A.Compose([A.LongestMaxSize(imsize),
                                     A.PadIfNeeded(imsize, imsize,
                                                   value=0, border_mode=cv2.BORDER_CONSTANT),
                                    #  A.Normalize(),
                                     ToTensorV2()])

    def _load_data(self):
        images_path = list(filter(lambda x: x.parent.name.split()[-2] in self._classes_config["all_classes_to_used"], self._data_path.glob("*/*")))
        return images_path

    def __len__(self):
        return len(self._items_path)

    def __getitem__(self, idx: int):
        item_path = self._items_path[idx]
        item_type_name = item_path.parent.name.split()[-2]
        item_class_name = self._classes_config["all_classes_to_used"][item_type_name]
        item_class_idx = self._classes_config["used_classes"][item_class_name]

        image = cv2.imread(str(item_path))

        if self._use_augs:
            image = self._transforms(image=image)["image"]

        if self._grayscale:
            image = A.to_gray(image)

        input_image = self._to_tensor(image=image)["image"]

        return dict(image=input_image, label=item_class_idx)
