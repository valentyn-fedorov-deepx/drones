from argparse import ArgumentParser
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch

from data import COCOFullDataset
from model import DetrDetectionModel

def parse_args():
    parser = ArgumentParser("Train script")
    parser.add_argument("--config-path", required=True,
                        help="Path to the config file")
    parser.add_argument("--logdir", type=Path, default="logs",
                        help="Path where the checkpoints and history will be saved")

    parser.add_argument('--full-dataset', action="store_true",
                        help="Use the full dataset and not part")

    parser.add_argument("--ckpt-path", default=None,
                        help="Path to the checkpoint to resume training")

    return parser.parse_args()

def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch], dim=0)
    labels = [item[1] for item in batch]

    return {
        'pixel_values': pixel_values,
        'labels': labels
    }
    
if __name__ == '__main__':
    
    train_images_path = "/nvme0n1-disk/stepan.severylov/datasets/for-detection/train"
    val_images_path = "/nvme0n1-disk/stepan.severylov/datasets/for-detection/valid"

    train_dataset = COCOFullDataset(train_images_path)
    val_dataset = COCOFullDataset(val_images_path)

    subset_indices = list(range(0, 1000))  
    subset_t = Subset(train_dataset, subset_indices)
    subset_v = Subset(val_dataset, subset_indices)
    
    model = DetrDetectionModel(n_classes=1, backbone='detr-resnet-50')

    train_dataloader = DataLoader(subset_t,
                                  batch_size=32,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(subset_v,
                                batch_size=32,
                                shuffle=True,
                                collate_fn=collate_fn)

    save_path = Path('results') / 'model_test'
    save_path.mkdir(parents=True, exist_ok=True)

    callbacks = [TQDMProgressBar(),
                 ModelCheckpoint(save_path,
                                 monitor="val_mAP_50",
                                 mode="max",
                                 save_last=True,
                                 save_top_k=2)
                 ]

    # hyperparameters = dict(batch_size=config["data"]["batch_size"],
    #                        image_size=config["data"]["image_size"],
    #                        model_name=config["model"]['model_name'],
    #                        backbone=config["model"]['backbone'],
    #                        encoder_frozen=config["model"]['encoder_freeze'],
    #                        pretrained=config["model"]['encoder_weights'],
    #                        loss=config["model"]['loss'],
    #                        **config["model"]["optimizer"])

    loggers = [
        # TensorBoardLogger(save_path, "tb_logs"),
        WandbLogger('detr_coco_20_epo',
                    save_path, project='DETR_object_detection')
    ]

    trainer = pl.Trainer(default_root_dir=save_path,
                        log_every_n_steps=5,
                        accelerator='gpu',
                        max_epochs=20,
                        callbacks=callbacks,
                        logger=loggers)

    # # for logger in loggers:
    # #     logger.log_hyperparams(hyperparameters)

    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    
    # for i, data in enumerate(train_dataloader):
    #     print(data['pixel_values'].shape)
    #     print(len(data['labels']))
        
    #     if i >= 2:
    #         break