from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from loguru import logger
from omegaconf import OmegaConf

from src.drone_pipeline.interfaces import BaseDetector, Detection
from src.utils.common import resource_path


class YoloV8Detector(BaseDetector):
    """Thin YOLOv8 wrapper that returns backend-agnostic detections.

    This class is intentionally *detection-only*: it does not embed any
    tracking or background-variance logic. That functionality must live in the
    dedicated tracking and pipeline modules.
    """

    def __init__(
        self,
        config_path: str,
        models_dir: str,
        device: str = "cuda",
        processed_labels: list[str] | None = None,
    ) -> None:
        cfg_path = resource_path(config_path)
        self.config = OmegaConf.load(cfg_path)

        # Optional physical prior for distance estimation
        self.drone_width = float(getattr(self.config, "drone_width_meters", 0.0))

        use_trt = bool(getattr(self.config, "use_tensorrt", False))
        trt_engine = str(getattr(self.config, "trt_engine", ""))
        if use_trt and trt_engine:
            model_path = os.path.join(models_dir, trt_engine)
        else:
            model_path = os.path.join(models_dir, self.config.model_name)
        model_path = resource_path(model_path)
        if use_trt and trt_engine and not os.path.exists(model_path):
            logger.warning(f"TensorRT engine not found at {model_path}, falling back to .pt")
            model_path = resource_path(os.path.join(models_dir, self.config.model_name))

        # Try to import ultralytics lazily so that the rest of the pipeline can
        # still run even if YOLOv8 is not available in the current environment.
        self.model: Optional[object]
        try:
            from ultralytics import YOLO  # type: ignore

            logger.info(f"Loading YOLOv8 model from {model_path}")
            self.model = YOLO(model_path)
            # Only PyTorch models support .to(device); exported formats manage device internally
            try:
                if str(model_path).endswith(".pt"):
                    self.model = self.model.to(device)
            except Exception as e:
                logger.warning(f"Failed to move model to {device}: {e}")
            self.device = device
        except ImportError as e:  # pragma: no cover - env dependent
            logger.error(
                "ultralytics is not installed; YoloV8Detector will return no detections. "
                "Install 'ultralytics' in a compatible environment to enable YOLOv8."
            )
            self.model = None
            self.device = device

        # Labels to keep (e.g. ["drone"]); None => keep everything
        self._processed_labels = processed_labels

        # Basic params from config
        self._imgsz = int(self.config.infer_imgsize)
        self._conf = float(self.config.model_conf)
        self._augment = bool(getattr(self.config, "model_augment", False))

    def get_drone_width(self) -> float:
        """Return configured drone width in meters (if available)."""
        return self.drone_width

    def get_conf(self) -> float:
        return float(self._conf)

    def get_imgsz(self) -> int:
        return int(self._imgsz)

    def detect_with_overrides(
        self,
        frame: np.ndarray,
        timestamp: float,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
        augment: Optional[bool] = None,
    ) -> List[Detection]:
        """Run YOLOv8 forward pass with optional runtime overrides."""

        # If YOLO model is not available, return no detections but keep the
        # pipeline running. This is useful on environments where ultralytics
        # wheels are not available (e.g., very new Python versions).
        if self.model is None:
            return []

        use_conf = self._conf if conf is None else float(conf)
        use_imgsz = self._imgsz if imgsz is None else int(imgsz)
        use_augment = self._augment if augment is None else bool(augment)

        # Ultralytics expects RGB or BGR; the current codebase typically works
        # in RGB for processing. We assume `frame` is RGB here, consistent with
        # existing usages of `view_img`.
        results = self.model(
            frame,
            imgsz=use_imgsz,
            conf=use_conf,
            augment=use_augment,
            verbose=False,
        )[0]

        names = results.names
        boxes = results.boxes

        detections: List[Detection] = []
        if boxes is None or boxes.shape[0] == 0:
            return detections

        for i in range(boxes.shape[0]):
            cls_idx = int(boxes.cls[i].item())
            label = names.get(cls_idx, str(cls_idx))
            if self._processed_labels is not None and label not in self._processed_labels:
                continue

            xyxy = boxes.xyxy[i].detach().cpu().numpy().astype(int)
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            score = float(boxes.conf[i].item())

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    score=score,
                    label=label,
                    timestamp=timestamp,
                )
            )

        logger.debug(f"YOLOv8: produced {len(detections)} detections")
        return detections

    def detect(self, frame: np.ndarray, timestamp: float) -> List[Detection]:  # type: ignore[override]
        """Run YOLOv8 forward pass and map boxes to Detection objects.

        This call is synchronous; the pipeline is responsible for scheduling it
        on a background thread so that the main video loop is never blocked.
        """
        return self.detect_with_overrides(frame, timestamp)
