from typing import List, Tuple

from src.cv_module.detected_object import DetectedObject
from src.cv_module.people.people_pose_classifier import PoseClassifier
# from src.cv_module.people.people_tracking import PeopleTracker
from src.cv_module.tracking.norfair_tracker import NorfairTracker
from src.cv_module.people.person import Person
from src.cv_module.people.people_range_estimation_v2 import PeopleRangeEstimator
from src.data_pixel_tensor import DataPixelTensor


class PeopleProcessorPipeline:
    def __init__(self, configs_dir, focal_distance_px: float,
                 imsize: Tuple[int]):
        self.imsize = imsize
        self._people_pose_classifier = PoseClassifier()

        self._people_tracker = NorfairTracker()

        self._people_range_estimation = PeopleRangeEstimator(focal_distance_px,
                                                             self.imsize)

    def process(self, detected_people: List[DetectedObject],
                data_pixel_tensor: DataPixelTensor) -> List[Person]:
        people = self._people_tracker.track(detected_people,
                                            data_pixel_tensor.view_img)

        people = self._people_pose_classifier.classify(people)

        people = self._people_range_estimation.set_distance(people)

        return people
