import yaml
import argparse
import torch
from pathlib import Path
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

from evaluate import update_confusion_matrix, compute_metrics
from model import segmentation_ffnet40S_BBB_mobile_pre_down
from dataset import CityscapesDataset


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def evaluate(model, dataloader, device, criterion, num_classes=19):
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation", unit="sample", leave=False):
            images = images.to(device)
            masks = masks.to(device).long()
            logits = model(images)
            # Upsample logits to match target mask size.
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)
            total_loss += loss.item()
            count += 1
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            for i in range(images.size(0)):
                gt = masks[i].cpu().numpy()
                confusion_matrix = update_confusion_matrix(confusion_matrix, gt, preds[i], num_classes)
    avg_loss = total_loss / count if count > 0 else 0.0
    metrics = compute_metrics(confusion_matrix)
    model.train()
    return metrics, confusion_matrix, avg_loss


# ======= Configuration Loading =======
def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train segmentation model on Cityscapes dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    config = load_config(args.config)

    # Hyperparameters from config
    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 4)
    learning_rate = config.get("learning_rate", 1e-3)
    weight_decay = config.get("weight_decay", 0)
    log_interval = config.get("log_interval", 50)
    checkpoint_interval = config.get("checkpoint_interval", 500)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Create model, loss, optimizer.
    model = segmentation_ffnet40S_BBB_mobile_pre_down(config.get("weights_path")).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Prepare datasets and dataloaders.
    train_dataset = CityscapesDataset(data_split="train", augment=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataset = CityscapesDataset(data_split="validation", augment=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Setup output directories using pathlib.
    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Setup TensorBoard writer.
    writer = SummaryWriter(log_dir=str(logs_dir))

    best_mean_iou = 0.0
    global_step = 0
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device).long()

            optimizer.zero_grad()
            logits = model(images)  # Output shape: [B, num_classes, H', W']
            # Upsample logits to match target mask size.
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
            writer.add_scalar("Train/Loss", loss.item(), global_step)

            # Log training images and predictions every log_interval iterations.
            if global_step % log_interval == 0:
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                # Log first image in the batch.
                input_img = images[0].detach().cpu()  # [C, H, W]
                # If grayscale, replicate to 3 channels.
                if input_img.shape[0] == 1:
                    input_img_vis = input_img.repeat(3, 1, 1)
                else:
                    input_img_vis = input_img
                # Normalize predicted mask for visualization.
                num_classes = 19
                pred_mask = preds[0].detach().cpu().unsqueeze(0).float() / (num_classes - 1)
                writer.add_image("Train/Input", input_img_vis, global_step)
                writer.add_image("Train/Prediction", pred_mask, global_step)

            # Save checkpoint every checkpoint_interval iterations.
            if global_step % checkpoint_interval == 0 and global_step > 0:
                ckpt_path = checkpoints_dir / f"checkpoint_{global_step}.pth"
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                }, str(ckpt_path))
            global_step += 1

        # ======= Validation at the end of each epoch =======
        val_metrics, val_conf_matrix, val_loss = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {epoch+1} Validation Metrics:")
        print(val_metrics)
        print(f"Validation Loss: {val_loss:.4f}")
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Overall_Accuracy", val_metrics["overall_accuracy"], epoch)
        writer.add_scalar("Val/Mean_Accuracy", val_metrics["mean_accuracy"], epoch)
        writer.add_scalar("Val/Mean_IoU", val_metrics["mean_iou"], epoch)
        writer.add_scalar("Val/FW_IoU", val_metrics["frequency_weighted_iou"], epoch)

        # Optionally, save the validation confusion matrix as an image.
        plt.figure(figsize=(8, 6))
        plt.imshow(val_conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Validation Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(val_conf_matrix.shape[0])
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        val_cm_path = checkpoints_dir / f"val_confusion_matrix_epoch_{epoch+1}.png"
        plt.savefig(str(val_cm_path))
        plt.close()

        # Save the best model based on mean IoU.
        if val_metrics["mean_iou"] > best_mean_iou:
            best_mean_iou = val_metrics["mean_iou"]
            best_model_path = checkpoints_dir / "best_model.pth"
            torch.save(model.state_dict(), str(best_model_path))
            print(f"New best model saved with Mean IoU: {best_mean_iou:.4f}")

    # Save final model checkpoint.
    final_ckpt = checkpoints_dir / "final_model.pth"
    torch.save(model.state_dict(), str(final_ckpt))
    writer.close()


if __name__ == "__main__":
    main()