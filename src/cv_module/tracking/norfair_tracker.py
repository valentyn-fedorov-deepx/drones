from norfair import Tracker, Detection
import numpy as np
from typing import List

from src.cv_module.basic_object import BasicObjectWithDistance
from src.cv_module.detected_object import DetectedObject
from src.cv_module import create_tracked_object_from_detected

from src.cv_module.people.person import Person


class NorfairTracker:
    def __init__(self):
        self.tracker = Tracker(distance_function="euclidean",
                               distance_threshold=1500, initialization_delay=0,
                               hit_counter_max=1, pointwise_hit_counter_max=4)

    def track(self, detected_objects: List[DetectedObject], img=None) -> List[BasicObjectWithDistance]:
        norfair_detections = [Detection(detected_object.get_tracking_points(),
                                        detected_object.get_tracking_points_conf(),
                                        data=detected_object, label=detected_object.name) for detected_object in detected_objects]

        tracked_objects = self.tracker.update(detections=norfair_detections)
        active_objects = list()
        for tracked_object in tracked_objects:
            new_object = create_tracked_object_from_detected(tracked_object.last_detection.data,
                                                             object_id=tracked_object.id)
            active_objects.append(new_object)

        # active_objects = [self.tracks[track.id] for track in self.tracker.get_active_objects()]
        return active_objects
