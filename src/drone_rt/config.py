from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from omegaconf import OmegaConf
from loguru import logger

from src.utils.common import resource_path


@dataclass
class RTDetectorConfig:
    """Config for lightweight realtime YOLO-based drone detector."""

    model_path: str
    imgsz: int = 512
    conf: float = 0.35
    iou: float = 0.5
    device: str = "cuda"
    labels: Optional[List[str]] = None

    @classmethod
    def from_yaml(cls, cfg_path: str) -> "RTDetectorConfig":
        cfg_path = resource_path(cfg_path)
        cfg = OmegaConf.load(cfg_path)

        model_path = cfg.get("model_path", None)
        if not model_path:
            raise ValueError(f"RTDetectorConfig: 'model_path' is required in {cfg_path}")

        labels_val = cfg.get("labels", None)
        labels: Optional[List[str]]
        if labels_val is None:
            labels = None
        else:
            # OmegaConf list -> python list[str]
            labels = [str(x) for x in labels_val]

        return cls(
            model_path=str(model_path),
            imgsz=int(cfg.get("imgsz", 512)),
            conf=float(cfg.get("conf", 0.35)),
            iou=float(cfg.get("iou", 0.5)),
            device=str(cfg.get("device", "cuda")),
            labels=labels,
        )


@dataclass
class RTTrackerConfig:
    """Config for lightweight IOU-based tracker."""

    max_age_frames: int = 15
    min_hits: int = 3
    iou_match_threshold: float = 0.3
    max_tracks: int = 8

    @classmethod
    def from_yaml(cls, cfg_path: str) -> "RTTrackerConfig":
        cfg_path = resource_path(cfg_path)
        try:
            cfg = OmegaConf.load(cfg_path)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"RTTrackerConfig: failed to load {cfg_path}, using defaults: {e}")
            return cls()

        return cls(
            max_age_frames=int(cfg.get("max_age_frames", 15)),
            min_hits=int(cfg.get("min_hits", 3)),
            iou_match_threshold=float(cfg.get("iou_match_threshold", 0.3)),
            max_tracks=int(cfg.get("max_tracks", 8)),
        )
