import numpy as np
from typing import List

from .car import Car
from src.cv_module.detected_object import DetectedObject

try:
    from norfair import Tracker, Detection  # type: ignore
    _HAS_NORFAIR = True
except Exception:
    Tracker = None  # type: ignore[assignment]
    Detection = None  # type: ignore[assignment]
    _HAS_NORFAIR = False


class CarTracker:
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

    def track(self, cars: List[DetectedObject]) -> List[Car]:
        if not cars:
            return []
        if not _HAS_NORFAIR or self.tracker is None:
            # Fallback: return empty list if norfair is unavailable
            return []

        norfair_detections = [
            Detection(
                car_detection.get_tracking_points(),
                car_detection.get_tracking_points_conf(),
                data=car_detection,
                label="Car",
            )
            for car_detection in cars
        ]

        tracked_objects = self.tracker.update(detections=norfair_detections)
        active_objects = []
        for tracked_object in tracked_objects:
            new_car = Car(
                id=tracked_object.id,
                detection=tracked_object.last_detection.data,
                bbox=np.array(tracked_object.last_detection.data.bbox),
                conf=tracked_object.last_detection.data.conf,
                mask=tracked_object.last_detection.data.mask,
            )
            active_objects.append(new_car)

        return active_objects
