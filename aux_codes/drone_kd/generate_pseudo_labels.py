from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(src_dir: Path) -> list[Path]:
    files: list[Path] = []
    for p in src_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    files.sort()
    return files


def generate_for_split(
    model: YOLO,
    src_dir: Path,
    out_root: Path,
    imgsz: int = 512,
    conf_thres: float = 0.35,
    label_name: str = "drone",
) -> None:
    """Run teacher model on all images in src_dir and save YOLO labels under out_root.

    out_root layout:
      out_root/images/*.jpg
      out_root/labels/*.txt
    """

    img_paths = collect_images(src_dir)
    if not img_paths:
        logger.warning(f"No images found in {src_dir}")
        return

    images_out = out_root / "images"
    labels_out = out_root / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(img_paths)} images from {src_dir} -> {out_root}")

    for img_path in tqdm(img_paths, desc=f"{src_dir.name}"):
        # Run teacher
        results = model(
            str(img_path),
            imgsz=imgsz,
            conf=conf_thres,
            verbose=False,
        )[0]

        boxes = results.boxes
        names = results.names
        lines: list[str] = []
        if boxes is not None and boxes.shape[0] > 0:
            for i in range(boxes.shape[0]):
                cls_idx = int(boxes.cls[i].item())
                label = names.get(cls_idx, str(cls_idx))
                if label_name is not None and label != label_name:
                    continue

                xywhn = boxes.xywhn[i].detach().cpu().numpy().astype(float)
                x_c, y_c, w, h = xywhn.tolist()

                # Single class dataset -> class_id = 0
                class_id = 0
                lines.append(
                    f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n"
                )

        # Copy image
        dst_img = images_out / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        # Write label file (empty file = background frame)
        label_path = labels_out / f"{img_path.stem}.txt"
        if lines:
            with label_path.open("w", encoding="utf-8") as f:
                f.writelines(lines)
        else:
            # Ensure an empty file exists so training knows this frame has no objects
            label_path.touch(exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate YOLO pseudo-labels for drone KD using YOLOv11x teacher.",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="models/drones_hf/yolov11x/weight/best.pt",
        help="Path to teacher model (.pt)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root folder with train/valid/test splits (Roboflow export)",
    )
    parser.add_argument(
        "--subset-name",
        type=str,
        default="drone",
        help="Name of subset folder inside each split (e.g. 'drone')",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="data_kd/drone",
        help="Output root for distilled dataset (Ultralytics layout)",
    )
    parser.add_argument("--imgsz", type=int, default=512, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--label-name", type=str, default="drone", help="Class name to keep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    teacher_path = Path(args.teacher)
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    logger.info(f"Loading teacher model from {teacher_path}")
    model = YOLO(str(teacher_path))

    # Splits we will process
    splits = ["train", "valid"]  # test можна додати окремо за потреби

    for split in splits:
        src_dir = data_root / split / args.subset_name
        if not src_dir.exists():
            logger.warning(f"Skip split {split}: {src_dir} does not exist")
            continue
        split_out = out_root / split
        generate_for_split(
            model=model,
            src_dir=src_dir,
            out_root=split_out,
            imgsz=args.imgsz,
            conf_thres=args.conf,
            label_name=args.label_name,
        )

    logger.info(f"Pseudo-label generation finished. Output root: {out_root}")


if __name__ == "__main__":
    main()
