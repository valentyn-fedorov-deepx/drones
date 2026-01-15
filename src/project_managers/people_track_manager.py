import time
from typing import Dict, List
import pickle as pkl

from omegaconf import OmegaConf
from loguru import logger

from src.data_pixel_tensor import DataPixelTensor
from src.cv_module.detected_object import DetectedObject
from src.project_managers.outputs.project_la_outputs import generate_response_dict
from src.cv_module.people.people_pose_estimation import PeoplePoseEstimator
from src.cv_module.detectors.yolo_detector import YoloDetector
from src.cv_module.people.people_processor import PeopleProcessorPipeline
from src.cv_module.visualization import plot_object_with_distance


class PeopleTrackManager:
    def __init__(self, config_path: str, device: str = "cuda"):
        self.config = OmegaConf.load(config_path)

        self.imsize = (self.config.im_height, self.config.im_width)
        self.focal_length_mm = self.config.focal_length_mm
        self.focal_length_px = self.focal_length_mm / self.config.sensor_ratio

        self.people_detector = YoloDetector(self.config.people_detection_config,
                                            self.config.people_detection_models_path,
                                            device, ["person"], refine_masks=True)
        self.people_pose_estimator = PeoplePoseEstimator(self.config.people_processing_config_path,
                                                         self.config.people_detection_models_path,
                                                         device=device)
        self.people_processor = PeopleProcessorPipeline(self.config.people_processing_config_path,
                                                        self.focal_length_px, self.imsize)

        self.detected_objects = list()
        self.latest_data_tensor = None

        self.longitude = self.config.longitude
        self.latitude = self.config.latitude
        self.altitude = self.config.altitude
        self.heading = self.config.heading

    def change_focal_length(self, new_focal_length):
        self.focal_length_mm = new_focal_length
        self.focal_length_px = self.focal_length_mm / self.config.sensor_ratio

        self.people_processor = PeopleProcessorPipeline(self.config.people_processing_config_path,
                                                        self.focal_length_px, self.imsize)

    def process_people(self, data_tensor: DataPixelTensor,
                       detected_people: List[DetectedObject]):
        start_time = time.time()
        detected_people = self.people_pose_estimator.process(data_tensor.view_img, detected_people)
        logger.debug(f"Pose estimation time: {time.time() - start_time:.3f} sec")

        start_time = time.time()
        tracked_people = self.people_processor.process(detected_people, data_tensor)
        logger.debug(f"People processing time: {time.time() - start_time:.3f} sec") 

        if tracked_people is None:
            return list()
        return tracked_people

    def process(self, data_tensor):
        start_time = time.time()

        data_tensor.calculate_all()
        data_tensor.convert_to_numpy()
        if data_tensor.n_xyz.dtype != 'uint8':
            data_tensor.convert_to_uint8()
        # import ipdb; ipdb.set_trace()
        logger.debug(f"Data tensor conversion time: {time.time() - start_time:.3f} sec")

        start_time = time.time()
        view_img = data_tensor["view_img"]
        detected_people = self.people_detector.predict(view_img)
        logger.debug(f"Detection time: {time.time() - start_time:.3f} sec")

        detected_people = self.process_people(data_tensor, detected_people)

        self.detected_objects = detected_people
        self.latest_data_tensor = data_tensor

        logger.debug(f"Processing {len(self.detected_objects)} objects")
        logger.debug(self.detected_objects)

    def generate_vizualization_for_latest_data(self):
        vizualization = self.latest_data_tensor.view_img.copy()
        # for person in self.detected_objects:
        vizualization = plot_object_with_distance(vizualization, self.detected_objects,
                                                  overlay_img=self.latest_data_tensor.n_xyz.copy())

        self.latest_data_tensor.visualized_data = vizualization
        return vizualization

    def get_latest_outputs(self, response_items: Dict):
        self.generate_vizualization_for_latest_data()

        response = generate_response_dict(self.latest_data_tensor, self.detected_objects,
                                          response_items)

        return response

    def dump_data(self, path: str):
        saved_data = dict()

        if self.detected_objects is not None:
            saved_data['detected_objects'] = self.detected_objects

        with open(path, 'wb') as f:
            pkl.dump(saved_data, f)


if __name__ == "__main__":
    pass
