from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from loguru import logger

from src.drone_pipeline.interfaces import TrackedObjectState

from .config import RTDetectorConfig, RTTrackerConfig
from .detector import DroneRTDetector
from .tracker import SimpleIOUTracker


@dataclass
class RTPipelineConfig:
    """High-level config for realtime drone pipeline.

    Attributes:
        detector_cfg_path: path to YAML with RTDetectorConfig.
        tracker_cfg_path: path to YAML with RTTrackerConfig.
        detection_interval: run detector every N frames (when tracking stable).
    """

    detector_cfg_path: str
    tracker_cfg_path: str
    detection_interval: int = 5


class DroneRTPipeline:
    """Minimal synchronous realtime pipeline: detect sometimes, track always.

    On each frame:
      * optionally run YOLO-based detector (every `detection_interval` frames);
      * feed detections to a lightweight IOU tracker;
      * return current active tracks.
    """

    def __init__(self, cfg: RTPipelineConfig) -> None:
        self._frame_idx: int = 0

        det_cfg = RTDetectorConfig.from_yaml(cfg.detector_cfg_path)
        trk_cfg = RTTrackerConfig.from_yaml(cfg.tracker_cfg_path)

        self._detector = DroneRTDetector(det_cfg)
        self._tracker = SimpleIOUTracker(trk_cfg)
        self._detection_interval = max(1, int(cfg.detection_interval))

    def process_frame(self, frame: np.ndarray, timestamp: float) -> List[TrackedObjectState]:
        """Process a single frame and return active tracks.

        This method is intentionally synchronous and lightweight. If later it
        becomes a bottleneck, only the detector call should be moved to a
        background thread.
        """

        self._frame_idx += 1

        do_detect = (self._frame_idx % self._detection_interval) == 1
        detections = []
        if do_detect:
            detections = self._detector.detect(frame, timestamp)
            logger.debug(f"[RTPipeline] frame={self._frame_idx} detections={len(detections)}")

        tracks = self._tracker.update(frame, timestamp, detections)
        return tracks
