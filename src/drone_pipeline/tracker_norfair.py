from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.cv_module.detected_object import DetectedObject
from src.cv_module.tracking.norfair_tracker import NorfairTracker
from src.drone_pipeline.interfaces import BaseTracker, Detection, TrackedObjectState


class NorfairTrackerWrapper(BaseTracker):
    """Adapter around the existing NorfairTracker.

    This wrapper respects the generic BaseTracker interface and returns
    backend-agnostic TrackedObjectState instances.
    """

    def __init__(self) -> None:
        self._tracker = NorfairTracker()

    @staticmethod
    def _detections_to_detected_objects(
        detections: List[Detection], frame_shape: tuple[int, int, int]
    ) -> List[DetectedObject]:
        H, W = frame_shape[:2]
        out: List[DetectedObject] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            # name is the detector label, usually "drone" here
            out.append(
                DetectedObject(
                    mask=None,
                    name=det.label,
                    conf=det.score,
                    imsize=(H, W),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    timestamp=det.timestamp,
                )
            )
        return out

    def update(
        self,
        frame: np.ndarray,
        timestamp: float,
        detections: Optional[List[Detection]] = None,
    ) -> List[TrackedObjectState]:  # type: ignore[override]
        H, W = frame.shape[:2]

        det_objs: List[DetectedObject]
        if detections:
            det_objs = self._detections_to_detected_objects(detections, frame.shape)
        else:
            det_objs = []

        # Underlying tracker maintains its own state over time
        tracked = self._tracker.track(det_objs, img=frame)

        states: List[TrackedObjectState] = []
        for obj in tracked:
            # obj is a BasicObjectWithDistance or subclass (Person/Car/etc.)
            x1, y1, x2, y2 = obj.bbox
            world_xyz = None
            velocity = None
            if getattr(obj, "meas", None) is not None:
                m = obj.meas
                world_xyz = (float(m.X), float(m.Y), float(m.Z))
                if getattr(m, "velocity", None) is not None:
                    vx, vy, vz = m.velocity
                    velocity = (float(vx), float(vy), float(vz))

            states.append(
                TrackedObjectState(
                    track_id=int(obj.id),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    score=float(getattr(obj, "conf", 1.0)),
                    label=str(getattr(obj, "name", "drone")),
                    timestamp=float(getattr(obj, "timestamp", timestamp)),
                    mask=getattr(obj, "mask", None),
                    world_xyz=world_xyz,
                    velocity_mps=velocity,
                )
            )

        return states
