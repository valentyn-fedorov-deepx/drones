import pytorch_lightning as pl
import timm
import torch
from torch.nn import CrossEntropyLoss
from typing import Union, Dict, Callable
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class ClassificationModel(pl.LightningModule):
    def __init__(self, model_name: str, n_classes: int, pretrained: bool = True,
                 optimizer: Union[Dict, str] = "adam", loss: Union[str, Callable] = "ce"):
        super().__init__()
        self._model = timm.create_model(model_name, pretrained=pretrained,
                                        num_classes=n_classes)
        self._loss = CrossEntropyLoss(reduction="mean")
        self._metrics = {
            "accuracy": MulticlassAccuracy(n_classes),
            "f1": MulticlassF1Score(n_classes)
        }

        self.optimizer = optimizer

        self.outputs = dict(train=list(),
                            test=list(),
                            val=list())

    def forward(self, image):
        # import ipdb; ipdb.set_trace()
        image = image / 255.
        # image -= torch.tensor(self._model.default_cfg['mean'])[:, None, None].to(image.device)
        # image /= torch.tensor(self._model.default_cfg['std'])[:, None, None].to(image.device)

        # import ipdb; ipdb.set_trace()

        prediction = self._model(image)
        return prediction

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch["image"])
        probs = torch.softmax(logits, dim=1)
        loss = self._loss(input=probs, target=batch["label"])

        metrics = {metric_name: metric_f.to(probs.device)(probs.argmax(1), batch["label"]) for metric_name, metric_f in self._metrics.items()}
        metrics["loss"] = loss
        self.log_dict(metrics, on_step=True, prog_bar=True)

        return metrics

    def on_validation_epoch_start(self):
        self.outputs["val"] = list()

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch["image"])
        probs = torch.softmax(logits, dim=1)

        loss = self._loss(input=probs, target=batch["label"])
        metrics = {f"val_{metric_name}": metric_f.to(probs.device)(probs.argmax(1), batch["label"]) for metric_name, metric_f in self._metrics.items()}
        metrics["val_loss"] = loss.item()
        self.outputs["val"].append(metrics)
        # self.log_dict(metrics, on_step=True)
        return metrics

    # def on_validation_epoch_end(self):
    #     stage = "val"
    #     self.log_dict(metrics, prog_bar=True)

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
