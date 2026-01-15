from pathlib import Path

from src.cv_module.people.people_pose_estimation import PeoplePoseEstimator
from src.cv_module.people.people_pose_classifier import PoseClassifier
from src.cv_module.people.people_segmentation import PeopleSegmenter
from src.cv_module.people.people_tracking import PeopleTracker
from src.cv_module.people.people_detection import YoloDetectorSlicing
from src.cv_module.people.person import Person
from src.cv_module.detectors.yolo_detector import YoloDetector
from src.data_pixel_tensor import DataPixelTensor


class PeopleDetectionPipeline:
    def __init__(self, configs_dir: str = "configs/people_detection",
                 models_dir: str = "models/people_detection",
                 device: str = "cpu",
                 track: bool = True,
                 segment: bool = False,
                 slicing_mode: bool = False):
        self.H, self.W = imsize = 2048, 2448
        self.configs_dir = Path(configs_dir)
        self.models_dir = Path(models_dir)
        if slicing_mode:
            self._people_detector = YoloDetectorSlicing(self.configs_dir,
                                                        models_dir, imsize, device=device)
        else:
            self._people_detector = YoloDetector(self.configs_dir / "yolo_detector.yaml",
                                                 models_dir, device=device,
                                                 processed_objects=['person'])

        self._people_pose_estimator = PeoplePoseEstimator(configs_dir,
                                                          models_dir,
                                                          device=device)
        self.segment = segment
        if segment:
            self._people_segmenter = PeopleSegmenter(configs_dir,
                                                     models_dir,
                                                     device=device)
        self._people_pose_classifier = PoseClassifier()

        self.track = track
        if track:
            self._people_tracker = PeopleTracker(configs_dir)

    def process(self, data_pixel_tensor: DataPixelTensor):
        detected_people = self._people_detector.predict(data_pixel_tensor.view_img)

        if self.track:
            people = self._people_tracker.track(detected_people,
                                                data_pixel_tensor.view_img)
        else:
            people = list()
            for detected_person in detected_people:
                person = Person(None, detected_person.bbox,
                                detected_person.conf, detected_person.mask,
                                kpts=None, moving=None)
                people.append(person)

        detected_people = self._people_pose_estimator.process(data_pixel_tensor.view_img,
                                                              detected_people)

        if self.segment:
            people = self._people_segmenter.process(data_pixel_tensor.view_img,
                                                    people)

        people = self._people_pose_classifier.classify(people)

        return people

    def clear_tracker(self):
        self._people_tracker = PeopleTracker(self.configs_dir)
