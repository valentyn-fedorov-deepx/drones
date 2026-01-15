import pytorch_lightning as pl
from transformers import DetrConfig, DetrForObjectDetection
import torch
import segmentation_models_pytorch as smp

from torchmetrics.detection.mean_ap import MeanAveragePrecision

class DetrDetectionModel(pl.LightningModule):

    def __init__(self, n_classes: int = 1, lr: int = 0.0001, 
                 backbone: str = 'detr-resnet-50'):
        super().__init__()

        self.model = DetrForObjectDetection.from_pretrained(f"facebook/{backbone}", 
                                                            num_labels=n_classes,
                                                            ignore_mismatched_sizes=True).train().to('cuda')

        self.lr = lr
        
        self.map_metric = MeanAveragePrecision(class_metrics=True, box_format='xywh', extended_summary=True)
        self.outputs = dict(train=list(),
                            test=list(),
                            val=list())

    def post_process_predictions(self, pred_boxes, logits):
        
        # scores shape: (batch_size, number of detected anchors, num_classes + 1) last class is the no-object class
        # pred_boxes shape: (batch_size, number of detected anchors, 4)

        predictions = []
        for score, box in zip(logits, pred_boxes):
            # Extract the bounding boxes, labels, and scores from the model's output
            pred_scores = score[:, :-1]  # Exclude the no-object class
            pred_boxes = box
            pred_labels = torch.argmax(pred_scores, dim=-1)

            # Get the scores corresponding to the predicted labels
            pred_scores_for_labels = torch.gather(pred_scores, 1, pred_labels.unsqueeze(-1)).squeeze(-1)
            predictions.append(
                {
                    "boxes": pred_boxes,
                    "scores": pred_scores_for_labels,
                    "labels": pred_labels,
                }
            )

        return predictions

    def forward(self, pixel_values, labels):
        outputs = self.model(pixel_values=pixel_values, labels=labels)
       
        return outputs
     
    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        
        pred_boxes = outputs.pred_boxes.detach()
        logits = outputs.logits.detach()
        
        preds = self.post_process_predictions(pred_boxes, logits)
        target = [{"boxes":t["boxes"], "labels":t["class_labels"]} for t in batch["labels"]]
                
        self.map_metric.update(preds, target)
        metrics = self.map_metric.compute()
                
        precision = metrics["precision"].mean()
        recall = metrics["recall"].mean()
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        
        return {
            "loss":loss,
            "loss_dict":loss_dict,
            "precision":precision,
            "recall":recall,
            "mAP_50":metrics['map_50']
        }

    def common_epoch_end(self, outputs, stage=""):
        loss = outputs["loss"]
        loss_dict = outputs["loss_dict"]
        precision = outputs["precision"]
        recall = outputs["recall"]
        map_50 = outputs["mAP_50"]
        
        return {
            f"{stage}_precision": precision.item(),
            f"{stage}_recall": recall.item(),
            f"{stage}_mAP_50": map_50.item(),
            f"{stage}_loss": loss.item(),
            f"{stage}_box_loss": loss_dict['loss_bbox'].item(),
        }
    
    def training_step(self, batch, batch_idx):
        results = self.common_step(batch, batch_idx)   
        metrics = self.common_epoch_end(results, "train")
        
        self.log_dict(metrics, on_step=True)
        
        return results

    def on_validation_epoch_start(self):
        self.outputs["val"] = list()

    def validation_step(self, batch, batch_idx):
        results = self.common_step(batch, batch_idx) 
        self.outputs["val"].append(results)   
        
        return results

    def on_validation_epoch_end(self):
        stage = "val"
        metrics = self.common_epoch_end(self.outputs[stage][0], stage)
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    model = DetrDetectionModel(20)
    print(model)
