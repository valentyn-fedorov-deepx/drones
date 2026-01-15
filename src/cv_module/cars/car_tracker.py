from norfair import Tracker, Detection
import numpy as np
from typing import List

from .car import Car
from src.cv_module.detected_object import DetectedObject


class CarTracker:
    def __init__(self):
        self.tracker = Tracker(distance_function="euclidean",
                               distance_threshold=1500, initialization_delay=0,
                               hit_counter_max=1, pointwise_hit_counter_max=4)

    def track(self, cars: List[DetectedObject]) -> List[Car]:
        norfair_detections = [Detection(car_detection.get_tracking_points(),
                                        car_detection.get_tracking_points_conf(),
                                        data=car_detection, label="Car") for car_detection in cars]

        tracked_objects = self.tracker.update(detections=norfair_detections)
        active_objects = list()
        for tracked_object in tracked_objects:
            new_car = Car(id=tracked_object.id,
                          detection=tracked_object.last_detection.data,
                          bbox=np.array(tracked_object.last_detection.data.bbox),
                          conf=tracked_object.last_detection.data.conf,
                          mask=tracked_object.last_detection.data.mask)
            active_objects.append(new_car)

        # active_objects = [self.tracks[track.id] for track in self.tracker.get_active_objects()]
        return active_objects
