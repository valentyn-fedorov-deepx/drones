import numpy as np
from typing import List

from src.cv_module.basic_object import BasicObjectWithDistance
from src.cv_module.detected_object import DetectedObject
from src.cv_module import create_tracked_object_from_detected

try:
    from norfair import Tracker, Detection  # type: ignore
    _HAS_NORFAIR = True
except Exception:
    Tracker = None  # type: ignore[assignment]
    Detection = None  # type: ignore[assignment]
    _HAS_NORFAIR = False


class NorfairTracker:
    def __init__(self):
        if _HAS_NORFAIR:
            self.tracker = Tracker(
                distance_function="euclidean",
                distance_threshold=1500,
                initialization_delay=0,
                hit_counter_max=1,
                pointwise_hit_counter_max=4,
            )
        else:
            self.tracker = None

    def track(self, detected_objects: List[DetectedObject], img=None) -> List[BasicObjectWithDistance]:
        if not detected_objects:
            return []

        if not _HAS_NORFAIR or self.tracker is None:
            # Fallback: keep a single stable ID (0) for the best detection
            best = max(detected_objects, key=lambda d: d.conf)
            return [create_tracked_object_from_detected(best, object_id=0)]

        norfair_detections = [
            Detection(
                detected_object.get_tracking_points(),
                detected_object.get_tracking_points_conf(),
                data=detected_object,
                label=detected_object.name,
            )
            for detected_object in detected_objects
        ]

        tracked_objects = self.tracker.update(detections=norfair_detections)
        active_objects = []
        for tracked_object in tracked_objects:
            new_object = create_tracked_object_from_detected(
                tracked_object.last_detection.data,
                object_id=tracked_object.id,
            )
            active_objects.append(new_object)

        return active_objects
