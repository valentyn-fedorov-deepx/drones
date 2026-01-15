from argparse import ArgumentParser
import json
from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from .model import ClassificationModel
from .dataset import StanfordCarClassificationDataset


def parse_args():
    parser = ArgumentParser("Train script")
    # parser.add_argument("--config-path", required=True,
    #                     help="Path to the config file")
    parser.add_argument("--logdir", type=Path, default="logs",
                        help="Path where the checkpoints and history will be saved")

    parser.add_argument('--full-dataset', action="store_true",
                        help="Use the full dataset and not part")

    parser.add_argument("--ckpt-path", default=None,
                        help="Path to the checkpoint to resume training")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_path = "/media/sviatoslav/MainVolume/Projects/deepxhub/data/car_data/car_data/train"
    val_path = "/media/sviatoslav/MainVolume/Projects/deepxhub/data/car_data/car_data/test"

    with open("data_config.json", "r") as f:
        data_config = json.load(f)

    train_dataset = StanfordCarClassificationDataset(train_path, data_config, use_augs=True)
    val_dataset = StanfordCarClassificationDataset(val_path, data_config, use_augs=False)

    model = ClassificationModel("efficientnet_b0", 3)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                  num_workers=8)
    val_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False,
                                num_workers=8)

    callbacks = [TQDMProgressBar(),
                 ModelCheckpoint(args.logdir,
                                 monitor="f1",
                                 mode="max",
                                 save_last=True,
                                 save_top_k=2)
                 ]
    loggers = [
        TensorBoardLogger(args.logdir, "tb_logs"),
        # WandbLogger(config["train"]["run_name"],
        #             save_path, project=config["train"]["logger"]["wandb_project"])
    ]

    trainer = pl.Trainer(default_root_dir=args.logdir,
                         #  fast_dev_run=2,
                         #  num_sanity_val_steps=0
                         #  **config["train"]["trainer"],
                         max_epochs=3,
                         log_every_n_steps=10,
                         callbacks=callbacks,
                         logger=loggers,
                         accelerator="cuda"
                         )

    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=args.ckpt_path)
