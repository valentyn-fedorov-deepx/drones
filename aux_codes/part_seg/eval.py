import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pathlib import Path
import yaml

from model import PartSegmentationModel
from data.data import PascalFullDataset, PascalPartInstanceCropSemanticDataset


def parse_args():
    parser = ArgumentParser("Script for calculating metrics on the dataset")

    parser.add_argument("--ckpt-path", type=Path,
                        help="Path to the checkpoint file")

    parser.add_argument("--split", default="val",
                        help="Name of the split from the config file")

    parser.add_argument("--full-dataset", default=False,
                        action="store_true",
                        help="Whether to use the full Pascal dataset")

    parser.add_argument("--save-path", type=Path,
                        default="logs_eval")

    parser.add_argument("--device", default="auto",
                        help="You can specify device that will be used. By default the script will try to use cuda")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    config_path = args.ckpt_path.parent / "config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    args.save_path.mkdir(parents=True, exist_ok=True)

    data_config = config["data"]
    dataset_config = data_config[args.split]

    if args.full_dataset:
        dataset = PascalFullDataset(**dataset_config)
    else:
        dataset = PascalPartInstanceCropSemanticDataset(**dataset_config,
                                                        labels_to_idx=data_config["labels_to_idx"],
                                                        labels_change_class=data_config["labels_change_class"])

    dataloader = DataLoader(dataset, batch_size=config["data"]["batch_size"])

    model = PartSegmentationModel.load_from_checkpoint(args.ckpt_path,
                                                       n_classes=dataset.n_classes,
                                                       **config["model"])

    device = args.device

    trainer = pl.Trainer(accelerator=device)

    output = trainer.validate(model, dataloader)

    with open(args.save_path / f"metrics_{config['train']['run_name']}.yaml", "w") as f:
        yaml.safe_dump(output, f)
