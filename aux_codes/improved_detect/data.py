import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import random
import os
from ultralytics.data.augment import Mosaic
from ultralytics.utils.ops import xyxy2xywhn, xywhn2xyxy
from ultralytics.utils.instance import Instances
from tqdm import tqdm


IMG_EXTS = ['.jpg', '.png', '.jpeg']


def create_our_augmentaitons(image_size):
    low_brightness = -0.1
    high_brightness = 0.2

    low_contrast = -0.1
    high_contrast = 0.1

    coco_augs = A.Compose([
        A.ToGray(always_apply=True),
        # A.Random,
        A.RandomCrop(image_size[0], image_size[1], always_apply=True),
        # A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(always_apply=False, p=0.3,
                                   brightness_limit=(low_brightness,
                                                     high_brightness),
                                   contrast_limit=(low_contrast,
                                                   high_contrast)),
        ], bbox_params=A.BboxParams(format='yolo',
                                    label_fields=["class_labels"],
                                    clip=False,
                                    min_width=0.3,
                                    min_height=0.5))

    return coco_augs


def create_cityperson_augmentations(image_size):
    low_brightness = -0.15
    high_brightness = 0.25

    low_contrast = -0.5
    high_contrast = 0.9

    cityperson_augs = A.Compose([
        A.ToGray(always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(always_apply=False, var_limit=(20, 100),
                     noise_scale_factor=0.3, p=0.3),
        A.PixelDropout(dropout_prob=0.1, always_apply=False,
                       p=0.3),
        A.RandomCropFromBorders(crop_left=0.4, crop_right=0.4, crop_top=0.4,
                                crop_bottom=0.4, p=0.2),
        A.RandomBrightnessContrast(always_apply=False, p=0.2,
                                   brightness_limit=(low_brightness,
                                                     high_brightness),
                                   contrast_limit=(low_contrast,
                                                   high_contrast)),
        A.AdvancedBlur(always_apply=False, p=0.2),
        A.LongestMaxSize(always_apply=True, max_size=max(image_size)),
        A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1],
                      border_mode=cv2.BORDER_CONSTANT,
                      value=0, always_apply=True)

    ], bbox_params=A.BboxParams(format='yolo',
                                label_fields=["class_labels"],
                                clip=True,
                                min_width=0.3,
                                min_height=0.5))

    return cityperson_augs


def create_basic_processsor(image_size):
    cityperson_augs = A.Compose([
        A.ToGray(always_apply=True),
        A.LongestMaxSize(always_apply=True, max_size=max(image_size)),
        A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1],
                      border_mode=cv2.BORDER_CONSTANT,
                      value=0, always_apply=True)
    ], bbox_params=A.BboxParams(format='yolo',
                                label_fields=["class_labels"],
                                clip=True,
                                min_width=0.3,
                                min_height=0.5))

    return cityperson_augs


def create_coco_augmentations(image_size):
    blur_limit = (5, 17)
    sigma_x_limit = (0.6, 1.1)
    sigma_y_limit = (0.6, 1.1)

    low_brightness = -0.15
    high_brightness = 0.25

    low_contrast = -0.5
    high_contrast = 0.9

    coco_augs = A.Compose([
        A.ToGray(always_apply=True),
        A.GaussNoise(always_apply=False, var_limit=(20, 100),
                     noise_scale_factor=0.3, p=0.2),
        A.PixelDropout(dropout_prob=0.1, always_apply=False,
                       p=0.2),
        # A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(always_apply=False, p=0.4,
                                   brightness_limit=(low_brightness, high_brightness),
                                   contrast_limit=(low_contrast, high_contrast)),
        A.AdvancedBlur(always_apply=False, p=0.3,
                       blur_limit=blur_limit,
                       sigma_x_limit=sigma_x_limit,
                       sigma_y_limit=sigma_y_limit,
                       rotate_limit=(-90, 90),
                       noise_limit=(0.1, 1.9)),
        A.LongestMaxSize(always_apply=True, max_size=max(image_size)),
        A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1],
                      border_mode=cv2.BORDER_CONSTANT,
                      value=0, always_apply=True)
    ], bbox_params=A.BboxParams(format='yolo',
                                label_fields=["class_labels"],
                                clip=True,
                                min_width=0.3,
                                min_height=0.5))

    return coco_augs


class ConcatDatasetsUltralytics(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.labels = []
        for dataset in self.datasets:
            self.labels += dataset.labels

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class UltralyticsFormatDataset(Dataset):
    def __init__(self, data_path: str, image_size: int = 512,
                 data_source: str = "coco", use_transforms: bool = True,
                 mosaic_prob: float = 0.0, fraction: float = 1.0):
        self._data_path = Path(data_path)
        self._mosaic_prob = mosaic_prob
        self._use_mosaic = mosaic_prob > 0.0

        self._images_path = Path(self._data_path / 'images')
        self._labels_path = Path(self._data_path / 'labels')

        self._use_transforms = use_transforms

        self._labels = []
        self._names = []

        self._image_size = image_size

        self._data_source = data_source
        if data_source == "coco":
            self._augmentations = create_coco_augmentations((self._image_size, self._image_size))
        elif data_source == 'citypersons':
            self._augmentations = create_cityperson_augmentations((self._image_size, self._image_size))
        elif data_source == 'our':
            self._augmentations = create_our_augmentaitons((self._image_size, self._image_size))
        else:
            self._augmentations = create_basic_processsor((self._image_size, self._image_size))

        if self._use_mosaic:
            self._mosaic_dataset = Mosaic(self, imgsz=self._image_size)

        for label in tqdm(os.listdir(self._labels_path)):
            file_label = []
            file_id = []

            with open(self._labels_path / label) as file:
                for line in file:
                    line = line.split()

                    class_id = int(line[0])

                    bbox = line[1:]
                    bbox = [float(i) for i in bbox]
                    bbox = torch.tensor(bbox, dtype=torch.float32)

                    if class_id == 0:
                        file_id.append(class_id)
                        file_label.append(bbox)

            if file_label:
                for file_ext in IMG_EXTS:
                    image_path = self._images_path / f"{label[:-4]}{file_ext}"

                    if image_path.exists():
                        self._labels.append({"cls": torch.tensor(file_id, dtype=torch.long),
                                            "bboxes": torch.stack(file_label, dim=0)})
                        self._names.append(label)
                        break

        self._fraction = fraction
        n_keep_labels = int(len(self._labels) * self._fraction)
        keep_idxs = np.random.choice(np.arange(len(self._labels), dtype=int),
                                     n_keep_labels, replace=False)
        self._labels = [self._labels[idx] for idx in keep_idxs]
        self._names = [self._names[idx] for idx in keep_idxs]

    @property
    def labels(self):
        return self._labels

    def prepare_bbox(self, size, labels):
        height, width = size
        class_ids = labels['cls']
        boxes_xywh = labels['bboxes']

        boxes_xyxy = torch.zeros_like(boxes_xywh)

        boxes_xywh[:, 0] *= width
        boxes_xywh[:, 1] *= height
        boxes_xywh[:, 2] *= width
        boxes_xywh[:, 3] *= height

        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch

    def _sample_mosaic(self, orig_sample, n: int):
        mix_labels = [orig_sample]

        while len(mix_labels) < n:
            sample_idx = random.randint(0, len(self) - 1)
            sample = self._process_item(sample_idx)
            if sample['bboxes'].nelement() == 0:
                continue
            mix_labels.append(sample)

        for im_labels in mix_labels:
            im_labels['instances'] = Instances(im_labels['bboxes'],
                                               bbox_format='xywh',
                                               normalized=True,
                                               segments=np.zeros((0, 100, 2),
                                                                 dtype=np.float32))
        mosaic_input = dict(rect_shape=None,
                            mix_labels=mix_labels)
        mosaic_input |= mix_labels[-1]
        mosaic_out = self._mosaic_dataset._mix_transform(mosaic_input)

        bboxes_in_correct_format = list()
        for bbox in mosaic_out['instances'].bboxes:
            bbox_correct = xyxy2xywhn(bbox, mosaic_out['img'].shape[1],
                                      mosaic_out['img'].shape[0])
            bboxes_in_correct_format.append(bbox_correct)

        mosaic_out['img'] = cv2.resize(mosaic_out['img'],
                                       (self._image_size, self._image_size))

        bboxes_in_correct_format = np.array(bboxes_in_correct_format)

        output = dict(im_file="",
                      ori_shape=mosaic_out['img'].shape[:2],
                      resized_shape=(self._image_size, self._image_size),
                      img=mosaic_out['img'],
                      cls=torch.tensor(mosaic_out['cls']),
                      bboxes=torch.tensor(bboxes_in_correct_format),
                      batch_idx=torch.zeros(len(mosaic_out['cls'])),
                      ratio_pad=torch.tensor((
                        [1, 1],  # gain, we divide bbox by this value
                        [0, 0])  # pad, and move by this value
                        # since we have our custom augmentations we do not need to specify them directly, at least for now
                        ))

        return output

    def _read_image(self, idx):
        image_path = None
        for file_ext in IMG_EXTS:
            image_path = self._images_path / f"{self._names[idx][:-4]}{file_ext}"
            if image_path.exists():
                break

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, image_path

    def _process_item(self, idx):
        label = self._labels[idx].copy()

        image, image_path = self._read_image(idx)
        image_orig_shape = image.shape[:2]

        if self._use_transforms:
            augmented = self._augmentations(image=image,
                                            bboxes=label['bboxes'].tolist(),
                                            class_labels=label['cls'].flatten().tolist())

            label['cls'] = torch.tensor(augmented['class_labels']).unsqueeze(1)
            label['bboxes'] = torch.tensor(augmented['bboxes'])
            image = augmented["image"]

        data = dict(im_file=str(image_path),
                    ori_shape=image_orig_shape,
                    resized_shape=(self._image_size, self._image_size),
                    img=image,
                    cls=label['cls'],
                    bboxes=label['bboxes'],
                    batch_idx=torch.zeros(len(label['cls'])),
                    ratio_pad=torch.tensor((
                        [1, 1],  # gain, we divide bbox by this value
                        [0, 0])  # pad, and move by this value
                        # since we have our custom augmentations we do not need to specify them directly, at least for now
                    ))

        return data

    def __getitem__(self, idx):
        data = self._process_item(idx)
        if self._use_mosaic:
            if random.random() < self._mosaic_prob:
                data = self._sample_mosaic(data, 4)

        image = ToTensorV2()(image=data['img'])['image']
        data['img'] = image

        return data

    def __len__(self):
        return len(self._labels)


if __name__ == "__main__":
    from ultralytics.data import build_dataloader
    from tqdm import tqdm
    import yaml
    from einops import rearrange

    data_config_path = "aux_codes/improved_detect/data.yaml"
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    mode = 'val'
    datasets = list()
    for i, dataset_info in enumerate(data_config['datasets']):
        dataset_path = Path(dataset_info['data_path']) / mode
        print('-------------------------------------')
        print(f"Processing {dataset_path}")
        print(dataset_info)
        split_dataset_info = dataset_info.copy()
        split_dataset_info['data_path'] = Path(dataset_info['data_path']) / mode

        dataset = UltralyticsFormatDataset(image_size=data_config['image_size'],
                                           **split_dataset_info,
                                           use_transforms=True)
        print(f"Dataset {i} size: {len(dataset)}")
        for item in dataset:
            pass
        datasets.append(dataset)
        # break

    full_dataset = ConcatDatasetsUltralytics(datasets)

    N = 8
    for i in range(N):
        idx = random.randint(0, len(full_dataset) - 1)
        data = full_dataset[idx]
        img = rearrange(data['img'].numpy(), 'c h w -> h w c')
        for bbox in data['bboxes']:
            x1, y1, x2, y2 = xywhn2xyxy(bbox, img.shape[1], img.shape[0]).round().numpy().astype(int).tolist()

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)

        cv2.imwrite(f'dataset_test_{idx}.jpg', img)

    dataloader = build_dataloader(full_dataset, 16,
                                  workers=1, shuffle=False)

    iter_dataloader = iter(dataloader)
    for i in range(100):
        batch = next(iter_dataloader)
