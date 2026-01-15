from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import torch_distributed_zero_first
from ultralytics.data import build_dataloader
from ultralytics.utils import IterableSimpleNamespace
from pathlib import Path
import yaml

from aux_codes.improved_detect.data import UltralyticsFormatDataset, ConcatDatasetsUltralytics


class CustomTrainer(DetectionTrainer):
    def get_dataset(self):
        with open(self.args.data, 'r') as f:
            self.data = yaml.safe_load(f)
        return self.args.data, self.args.data

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP

            datasets = list()
            with open(dataset_path, 'r') as f:
                data_config = yaml.safe_load(f)
            for dataset_info in data_config['datasets']:
                split_dataset_info = dataset_info.copy()
                split_dataset_info['data_path'] = Path(dataset_info['data_path']) / mode
                if mode == 'val':
                    split_dataset_info['mosaic_prob'] = False
                    split_dataset_info['data_source'] = None

                dataset = UltralyticsFormatDataset(image_size=data_config['image_size'],
                                                   fraction=self.args.fraction, **split_dataset_info)

                datasets.append(dataset)

            dataset = ConcatDatasetsUltralytics(datasets)

        shuffle = mode == "train"
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)


if __name__ == "__main__":
    with open('aux_codes/improved_detect/train.yaml', 'r') as f:
        train_cfg = yaml.safe_load(f)

    overrides = dict(
                     data="aux_codes/improved_detect/data.yaml",
                     #  data="/sdb-disk/vyzai/datasets/citypersons_2/data.yaml",
                     #  data="/sdb-disk/vyzai/datasets/coco_people_ultralytics/data.yaml",
                     model="yolo11m.pt",
                     #  fraction=0.1,
                     augment=False,
                     epochs=40,
                     lr0=1e-4,
                     lrf=0.01,
                     optimizer="SGD",
                     warmup_epochs=1,
                     #  warmup_momentum=0.8,
                     name='custom_trainer_yolov11m_coco_smaller_lr_full',
                     batch=32,
                     single_cls=True,
                     freeze=None
                     )

    # trainer = CustomTrainer(IterableSimpleNamespace(**train_cfg),
    #                         overrides=overrides)
    trainer = DetectionTrainer(IterableSimpleNamespace(**train_cfg),
                               overrides=overrides)

    trainer.train()
    # model = YOLO("yolov8m.pt", 'detect')
    # model.train(DetectionTrainer, **train_cfg)
