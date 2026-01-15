from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import cv2
from loguru import logger

from src.drone_pipeline.interfaces import BaseSegmenter, Detection
from src.utils.common import resource_path


@dataclass
class SamConfig:
    checkpoint: str
    model_type: str = "vit_l"
    device: str = "cuda"


class SamBBoxSegmenter(BaseSegmenter):
    """SAM-based segmenter using the detection bbox as a prompt.

    Primary mask-refinement backend. Produces bbox-local binary masks that
    align with the provided detection bbox.
    """

    def __init__(self, cfg: SamConfig) -> None:
        try:
            from segment_anything import SamPredictor, sam_model_registry  # type: ignore
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError(
                "segment_anything is not installed. Install it to use SamBBoxSegmenter."
            ) from e

        ckpt_path = resource_path(cfg.checkpoint)
        logger.info(f"Loading SAM from {ckpt_path} (type={cfg.model_type})")
        model = sam_model_registry[cfg.model_type](checkpoint=ckpt_path)
        self._predictor = SamPredictor(model.to(cfg.device))

    def segment(self, frame: np.ndarray, detection: Detection) -> np.ndarray:  # type: ignore[override]
        # SAM expects RGB; `frame` is assumed RGB consistent with the pipeline.
        self._predictor.set_image(frame)

        x1, y1, x2, y2 = detection.bbox
        box = np.array([[x1, y1, x2, y2]], dtype=np.float32)

        masks, scores, _ = self._predictor.predict(
            box=box,
            multimask_output=False,
        )

        if masks is None or len(masks) == 0:
            logger.warning("SAM returned no masks; falling back to empty mask")
            h = max(1, y2 - y1)
            w = max(1, x2 - x1)
            return np.zeros((h, w), dtype=np.uint8)

        full_mask = masks[0].astype(bool)  # H×W
        # Crop to bbox and convert to uint8 0/1
        h = max(1, y2 - y1)
        w = max(1, x2 - x1)
        crop = full_mask[max(0, y1):y2, max(0, x1):x2]
        if crop.shape[0] != h or crop.shape[1] != w:
            crop = cv2.resize(crop.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        return crop.astype(np.uint8)


class YoloV8SegFallback(BaseSegmenter):
    """YOLOv8-Seg based segmenter used as a fallback when SAM is unavailable.

    Strategy: run YOLOv8-Seg on the full frame and pick the mask whose bbox has
    the highest IoU with the requested detection bbox.
    """

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        from ultralytics import YOLO  # lazy import to avoid hard dependency at import time

        model_path = resource_path(model_path)
        logger.info(f"Loading YOLOv8-Seg model from {model_path}")
        self._model = YOLO(model_path).to(device)

    @staticmethod
    def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return float(inter / max(area_a + area_b - inter, 1e-6))

    def segment(self, frame: np.ndarray, detection: Detection) -> np.ndarray:  # type: ignore[override]
        results = self._model(frame, verbose=False)[0]
        if results.masks is None or results.boxes is None or results.boxes.shape[0] == 0:
            x1, y1, x2, y2 = detection.bbox
            h = max(1, y2 - y1)
            w = max(1, x2 - x1)
            return np.zeros((h, w), dtype=np.uint8)

        det_bbox = np.array(detection.bbox, dtype=float)

        # Find best matching segmentation by IoU
        best_idx: Optional[int] = None
        best_iou = 0.0
        for i in range(results.boxes.shape[0]):
            box = results.boxes.xyxy[i].detach().cpu().numpy().astype(float)
            iou = self._bbox_iou(det_bbox, box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx is None or best_iou == 0.0:
            x1, y1, x2, y2 = detection.bbox
            h = max(1, y2 - y1)
            w = max(1, x2 - x1)
            return np.zeros((h, w), dtype=np.uint8)

        # Retrieve mask and crop to detection bbox
        mask_prob = results.masks.data[best_idx].detach().cpu().numpy()  # Hm×Wm in [0,1]
        Hm, Wm = mask_prob.shape

        # Use normalized bbox coordinates if available for more accurate crop
        if hasattr(results.boxes, "xyxyn"):
            x1n, y1n, x2n, y2n = results.boxes.xyxyn[best_idx].detach().cpu().numpy()
            x1_m = int(x1n * Wm)
            y1_m = int(y1n * Hm)
            x2_m = int(x2n * Wm)
            y2_m = int(y2n * Hm)
        else:
            # Fallback: scale by image size
            H, W = frame.shape[:2]
            x1i, y1i, x2i, y2i = detection.bbox
            x1_m = int(x1i / max(W, 1) * Wm)
            x2_m = int(x2i / max(W, 1) * Wm)
            y1_m = int(y1i / max(H, 1) * Hm)
            y2_m = int(y2i / max(H, 1) * Hm)

        mask_crop = mask_prob[max(0, y1_m):y2_m, max(0, x1_m):x2_m] > 0.5
        x1, y1, x2, y2 = detection.bbox
        h = max(1, y2 - y1)
        w = max(1, x2 - x1)
        if mask_crop.shape[0] != h or mask_crop.shape[1] != w:
            mask_crop = cv2.resize(mask_crop.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        return mask_crop.astype(np.uint8)


class HybridSegmenter(BaseSegmenter):
    """Primary SAM + secondary YOLOv8-Seg fallback segmenter.

    Tries SAM first; on any failure, gracefully falls back to YOLOv8-Seg.
    """

    def __init__(
        self,
        sam: Optional[SamBBoxSegmenter] = None,
        yolo_fallback: Optional[YoloV8SegFallback] = None,
    ) -> None:
        # If both backends are None, the segmenter will gracefully return
        # an empty mask. This lets the rest of the pipeline run even when
        # SAM / YOLOv8-Seg are not available yet.
        self._sam = sam
        self._yolo = yolo_fallback

    def segment(self, frame: np.ndarray, detection: Detection) -> np.ndarray:  # type: ignore[override]
        # Try SAM first
        if self._sam is not None:
            try:
                return self._sam.segment(frame, detection)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"SAM segmenter failed, falling back to YOLOv8-Seg: {e}")

        if self._yolo is not None:
            try:
                return self._yolo.segment(frame, detection)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"YOLOv8-Seg segmenter failed, falling back to empty mask: {e}")

        # Fallback: empty bbox-local mask so the rest of the pipeline can run
        x1, y1, x2, y2 = detection.bbox
        h = max(1, y2 - y1)
        w = max(1, x2 - x1)
        return np.zeros((h, w), dtype=np.uint8)
