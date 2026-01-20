from __future__ import annotations

from ultralytics import YOLO


if __name__ == "__main__":
    # Student model (Ultralytics will download yolo11s.pt if not present)
    model = YOLO("yolo11s.pt")

    model.train(
        data="configs/project_drones/drone_kd_data.yaml",
        imgsz=512,
        epochs=80,
        batch=16,
        device="cuda",
        name="drone_yolo11s_kd",
        single_cls=True,
    )
