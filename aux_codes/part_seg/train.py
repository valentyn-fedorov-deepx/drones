from argparse import ArgumentParser
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from model import PartSegmentationModel
from data.data import PascalPartInstanceCropSemanticDataset, PascalFullDataset


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


if __name__ == '__main__':
    args = parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_config = config["data"]

    if args.full_dataset:
        train_dataset = PascalFullDataset(**data_config["train"])

        val_dataset = PascalFullDataset(**data_config["val"])
    else:
        train_dataset = PascalPartInstanceCropSemanticDataset(**data_config["train"],
                                                              labels_to_idx=data_config["labels_to_idx"],
                                                              labels_change_class=data_config["labels_change_class"])

        val_dataset = PascalPartInstanceCropSemanticDataset(**data_config["val"],
                                                            labels_to_idx=data_config["labels_to_idx"],
                                                            labels_change_class=data_config["labels_change_class"])

    model_config = config["model"]

    model = PartSegmentationModel(data_config["n_classes"],
                                  **model_config)

С
    val_dataloader = DataLoader(val_dataset,
                                batch_size=data_config["batch_size"])

    save_path = args.logdir / config["train"]["run_name"]
    save_path.mkdir(parents=True, exist_ok=True)

    callbacks = [TQDMProgressBar(),
                 ModelCheckpoint(save_path,
                                 monitor="val_dataset_iou",
                                 mode="max",
                                 save_last=True,
                                 save_top_k=2)
                 ]

    hyperparameters = dict(batch_size=config["data"]["batch_size"],
                           image_size=config["data"]["image_size"],
                           model_name=config["model"]['model_name'],
                           backbone=config["model"]['backbone'],
                           encoder_frozen=config["model"]['encoder_freeze'],
                           pretrained=config["model"]['encoder_weights'],
                           loss=config["model"]['loss'],
                           **config["model"]["optimizer"])

    loggers = [
        TensorBoardLogger(save_path, "tb_logs"),
        WandbLogger(config["train"]["run_name"],
                    save_path, project=config["train"]["logger"]["wandb_project"])
    ]

    trainer = pl.Trainer(default_root_dir=save_path,
                         #  fast_dev_run=2,
                         #  num_sanity_val_steps=0
                         **config["train"]["trainer"],
                         callbacks=callbacks,
                         logger=loggers
                         )

    for logger in loggers:
        logger.log_hyperparams(hyperparameters)

    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=args.ckpt_path)

    save_config_path = save_path / "config.yaml"
    with open(save_config_path, 'w') as f:
        yaml.safe_dump(config, f)
