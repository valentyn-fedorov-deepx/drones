import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from dataset import CityscapesDataset
from model import segmentation_ffnet40S_BBB_mobile_pre_down

# Assuming the CityscapesDataset is defined in the same script or imported accordingly
# from your_dataset_module import CityscapesDataset


# Dummy get_predictions function.
def get_predictions(ffnet, input_image: torch.Tensor) -> torch.Tensor:
    """
    Dummy prediction function.
    In production, replace this with the actual model inference.
    It should return logits with shape [num_classes, H, W].
    """
    # num_classes = 19  # Number of valid classes (ignoring those with trainId==255)
    _, H, W = input_image.shape
    prediction_logits = ffnet(input_image.unsqueeze(0))
    prediction = prediction_logits.argmax(1).unsqueeze(0).float()
    resized_prediction = F.interpolate(prediction, size=(H, W),
                                       mode='nearest').squeeze(0).long()

    return resized_prediction


def update_confusion_matrix(conf_matrix, gt, pred, num_classes=19):
    """
    Update confusion matrix given ground truth and predicted masks.
    Both gt and pred should be 2D numpy arrays of shape [H, W].
    Pixels with gt value 255 are ignored.
    """
    valid_mask = (gt != 255)
    gt_valid = gt[valid_mask].flatten()
    pred_valid = pred[valid_mask].flatten()

    if gt_valid.size == 0:
        return conf_matrix

    cur_cm = np.bincount(
        num_classes * gt_valid + pred_valid,
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    conf_matrix += cur_cm
    return conf_matrix


def compute_metrics(conf_matrix):
    """
    Compute evaluation metrics from the confusion matrix.
    Returns a dictionary of metrics.
    """
    overall_acc = np.diag(conf_matrix).sum() / conf_matrix.sum()

    class_acc = np.diag(conf_matrix) / (conf_matrix.sum(axis=1) + 1e-6)
    mean_acc = np.mean(class_acc)

    intersection = np.diag(conf_matrix)
    union = conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    iou = intersection / (union + 1e-6)
    mean_iou = np.mean(iou)

    freq = conf_matrix.sum(axis=1) / conf_matrix.sum()
    fw_iou = (freq[freq > 0] * iou[freq > 0]).sum()

    metrics = {
        "overall_accuracy": overall_acc,
        "mean_accuracy": mean_acc,
        "mean_iou": mean_iou,
        "frequency_weighted_iou": fw_iou,
        "class_accuracy": class_acc.tolist(),
        "iou": iou.tolist(),
    }
    return metrics


def main():
    weights_path = "/nvme0n1-disk/sviatoslav.darmohrai/vyzai/dx_vyzai_people_track/aux_codes/cityscapes_segmentation/weights/ffnet40S/ffnet40S_BBB_cityscapes_state_dict_quarts.pth"
    ffnet = segmentation_ffnet40S_BBB_mobile_pre_down(weights_path)
    # Create the validation dataset without augmentation.
    dataset = CityscapesDataset(data_split="validation", augment=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    num_classes = 19  # Evaluate on valid classes (trainId 0 to 18)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Process the validation set.
    for image, mask in tqdm(dataloader, desc="Validation", unit="sample"):
        # Remove the batch dimension.
        image = image.squeeze(0)  # shape: [C, H, W]
        pred = get_predictions(ffnet, image).cpu().numpy().squeeze(0)

        gt = mask.squeeze(0).numpy()
        # import ipdb; ipdb.set_trace()
        confusion_matrix = update_confusion_matrix(confusion_matrix, gt, pred, num_classes)

    # Compute metrics.
    metrics = compute_metrics(confusion_matrix)
    print("Evaluation Metrics:")
    print(json.dumps(metrics, indent=4))

    # Use pathlib to handle file paths.
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix as CSV.
    csv_path = metrics_dir / "confusion_matrix.csv"
    np.savetxt(str(csv_path), confusion_matrix, delimiter=",", fmt="%d")

    # Save metrics as JSON.
    json_path = metrics_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Save confusion matrix as an image.
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(confusion_matrix.shape[0])
    plt.xticks(tick_marks, tick_marks, rotation=45)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    img_path = metrics_dir / "confusion_matrix.png"
    plt.savefig(str(img_path))
    plt.close()


if __name__ == "__main__":
    main()
