import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from typing import Union, Dict, Callable
import numpy as np


class PartSegmentationModel(pl.LightningModule):
    def __init__(self, n_classes: int, model_name: str = 'unet', backbone: str = 'resnext50_32x4d',
                 encoder_weights: str = 'imagenet', encoder_freeze: bool = True,
                 optimizer: Union[Dict, str] = 'Adam', loss: Union[str, Callable] = 'focal'):
        super().__init__()
        self.model = smp.create_model(arch=model_name, encoder_name=backbone,
                                      in_channels=3, classes=n_classes,
                                      encoder_weights=encoder_weights)

        if encoder_freeze:
            for parameter in self.model.encoder.parameters():
                parameter.requires_grad = False

        self._n_classes = n_classes
        params = smp.encoders.get_preprocessing_params(backbone)
        self.register_buffer("std",
                             torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean",
                             torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.optimizer = optimizer

        if isinstance(loss, str):
            if loss == 'dice':
                self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, 
                                                   from_logits=True)
            elif loss == 'focal':
                self.loss_fn = smp.losses.FocalLoss(mode='multiclass')
            elif loss == 'cross_entropy':
                self.loss_fn = smp.losses.SoftCrossEntropyLoss()
            else:
                raise ValueError("Unknown loss")
        else:
            self.loss_fn = loss

        self.outputs = dict(train=list(),
                            test=list(),
                            val=list())

    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch):
        image = batch["image"]
        assert image.ndim == 4

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
        pred_mask = logits_mask.argmax(1)

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(),
                                               mode="multiclass",
                                               num_classes=self._n_classes)

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage=""):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss = np.mean([x["loss"].item() for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")

        return {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_f1": f1,
            f"{stage}_precision": precision,
            f"{stage}_recall": recall,
            f"{stage}_loss": loss,
        }

    def training_step(self, batch, batch_idx):
        results = self.shared_step(batch)
        metrics = self.shared_epoch_end([results], "train")
        self.log_dict(metrics, on_step=True)

        return results

    def on_validation_epoch_start(self):
        self.outputs["val"] = list()

    def validation_step(self, batch, batch_idx):
        results = self.shared_step(batch)

        self.outputs["val"].append(results)
        return results

    def on_validation_epoch_end(self):
        stage = "val"
        metrics = self.shared_epoch_end(self.outputs[stage], stage)
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        if isinstance(self.optimizer, str):
            if self.optimizer == 'adam':
                return torch.optim.Adam(self.parameters(), lr=0.0001)
        elif isinstance(self.optimizer, dict):
            if self.optimizer['name'] == 'adam':
                return torch.optim.Adam(self.parameters(),
                                        **self.optimizer['params'])
        else:
            raise ValueError("Incorrect optimizer config")


if __name__ == "__main__":
    model = PartSegmentationModel(20)
    print(model)
