from __future__ import annotations

from typing import List, Optional

import numpy as np
from loguru import logger

from ultralytics import YOLO

from src.drone_pipeline.interfaces import Detection
from src.utils.common import resource_path

from .config import RTDetectorConfig


class DroneRTDetector:
    """Thin YOLO wrapper for realtime drone detection.

    This detector is intentionally simple: one forward pass per call, no tiling,
    no background-gradient hacks, no internal tracking. It just produces
    backend-agnostic `Detection` objects.
    """

    def __init__(self, config: RTDetectorConfig) -> None:
        self._cfg = config

        model_path = resource_path(self._cfg.model_path)
        logger.info(f"[RTDetector] Loading YOLO model from {model_path}")

        self._model = YOLO(model_path)
        # Для .pt можна викликати .to(device); інші backends самі керують девайсом.
        try:
            if model_path.endswith(".pt"):
                self._model.to(self._cfg.device)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"[RTDetector] Failed to move model to {self._cfg.device}: {e}")

        self._label_whitelist: Optional[set[str]]
        if self._cfg.labels is None:
            self._label_whitelist = None
        else:
            self._label_whitelist = {str(l) for l in self._cfg.labels}

    def detect(self, frame: np.ndarray, timestamp: float) -> List[Detection]:
        """Run a single YOLO forward pass and map boxes to Detection objects."""

        if frame is None or frame.size == 0:
            return []

        # Ultralytics очікує RGB/BGR uint8; у нашому пайплайні `view_img` уже RGB.
        results = self._model(
            frame,
            imgsz=self._cfg.imgsz,
            conf=self._cfg.conf,
            iou=self._cfg.iou,
            verbose=False,
        )[0]

        boxes = results.boxes
        names = results.names
        if boxes is None or boxes.shape[0] == 0:
            return []

        dets: List[Detection] = []
        for i in range(boxes.shape[0]):
            cls_idx = int(boxes.cls[i].item())
            label = names.get(cls_idx, str(cls_idx))

            if self._label_whitelist is not None and label not in self._label_whitelist:
                continue

            xyxy = boxes.xyxy[i].detach().cpu().numpy().astype(int)
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            score = float(boxes.conf[i].item())

            dets.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    score=score,
                    label=label,
                    timestamp=timestamp,
                )
            )

        logger.debug(f"[RTDetector] produced {len(dets)} detections")
        return dets
