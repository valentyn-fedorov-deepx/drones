from omegaconf import OmegaConf
from typing import Dict, List
from loguru import logger

from src.data_pixel_tensor import DataPixelTensor
from src.cv_module.detected_object import DetectedObject
from src.project_managers.outputs.project_la_outputs import generate_response_dict
from src.cv_module.people.people_pose_estimation import PeoplePoseEstimator
from src.cv_module.detectors.yolo_detector import YoloDetectorTiledContextedTracked
from src.cv_module.cars.car_tracker import CarTracker
from src.cv_module.cars.car_distance import CarRangeEstimation
from src.cv_module.people.people_processor import PeopleProcessorPipeline
from src.cv_module.tracking.norfair_tracker import NorfairTracker
from src.cv_module.distance_measurers.width_measurer import WidthDistanceMeasurer
from src.cv_module.visualization import plot_object_with_distance
from src.utils.common import resource_path


class ProjectLAManager:
    def __init__(self, config_path: str, device: str = "cuda"):
        self.config = OmegaConf.load(resource_path(config_path))

        self.imsize = (self.config.im_height, self.config.im_width)
        self.focal_length_mm = self.config.focal_length_mm
        self.focal_length_px = self.focal_length_mm / self.config.sensor_ratio

        cfg_path = resource_path(self.config.drone_detection_config)
        model_path = resource_path(self.config.drone_detection_models_path)
        self.drones_detector = YoloDetectorTiledContextedTracked(
            cfg_path,
            model_path,
            device=device,
            processed_objects=["drone"],
        )

        # self.drones_detector = YoloDetector(self.config.drone_detection_config,
        #                                     self.config.drone_detection_models_path,
        #                                     device=device, processed_objects=None)

        # self.people_and_cars_detector = YoloDetector(self.config.people_cars_detection_config,
        #                                              self.config.people_cars_detection_models_path,
        #                                              device, ["person", "car", "truck"],
        #                                              name_map=dict(truck="car"))
        # self.people_pose_estimator = PeoplePoseEstimator(self.config.people_processing_config_path,
        #                                                  self.config.people_cars_detection_models_path,
        #                                                  device=device)
        # self.people_processor = PeopleProcessorPipeline(self.config.people_processing_config_path,
        #                                                 self.focal_length_px, self.imsize)

        # self.cars_tracker = CarTracker()
        self.cars_range_estimation = CarRangeEstimation(self.focal_length_px, self.imsize)

        self.drones_tracker = NorfairTracker()
        self.drones_distance_measurer = WidthDistanceMeasurer(
            self.focal_length_px, self.imsize,
            base_width_in_meters=self.drones_detector.get_drone_width()
        )

        self.detected_objects = list()
        self.latest_data_tensor = None

        self.longitude = self.config.longitude
        self.latitude = self.config.latitude
        self.altitude = self.config.altitude
        self.heading = self.config.heading

    def change_focal_length(self, new_focal_length):
        self.focal_length_mm = new_focal_length
        self.focal_length_px = self.focal_length_mm / self.config.sensor_ratio

        # self.people_processor = PeopleProcessorPipeline(self.config.people_processing_config_path,
        #                                                 self.focal_length_px, self.imsize)
        # self.cars_range_estimation = CarRangeEstimation(self.focal_length_px, self.imsize)
        self.drones_distance_measurer = WidthDistanceMeasurer(self.focal_length_px, self.imsize,
                                                              base_width_in_meters=0.221)

    def process_people(self, data_tensor: DataPixelTensor,
                       detected_people: List[DetectedObject]):
        detected_people = self.people_pose_estimator.process(data_tensor.view_img, detected_people)
        tracked_people = self.people_processor.process(detected_people, data_tensor)

        if tracked_people is None:
            return list()
        return tracked_people

    def process_cars(self, data_tensor: DataPixelTensor,
                     detected_cars: List[DetectedObject]):
        tracked_cars = self.cars_tracker.track(detected_cars)
        tracked_cars = self.cars_range_estimation.process(data_tensor.view_img,
                                                          tracked_cars)

        if tracked_cars is None:
            tracked_cars = list()
        return tracked_cars

    def process_drones(self, detected_drones: List[DetectedObject]):
        tracked_drones = self.drones_tracker.track(detected_drones)
        measurements = self.drones_distance_measurer.process(tracked_drones)

        for drone in tracked_drones:
            drone.meas = measurements[drone.id]

        return tracked_drones

    def process(self, data_tensor):
        # detected_people_and_cars = self.people_and_cars_detector.predict(data_tensor["view_img"])

        # detected_people = list()
        # detected_cars = list()
        # for detected_object in detected_people_and_cars:
        #     if detected_object.name == "person":
        #         detected_people.append(detected_object)
        #     elif detected_object.name == 'car':
        #         detected_cars.append(detected_object)
        detected_drones = self.drones_detector.predict(data_tensor["view_img"], data_tensor.created_at)

        # detected_people = self.process_people(data_tensor, detected_people)
        # detected_cars = self.process_cars(data_tensor, detected_cars)
        detected_people = []
        detected_cars = []

        detected_drones = self.process_drones(detected_drones)

        self.detected_objects = detected_people + detected_cars + detected_drones
        self.latest_data_tensor = data_tensor

        for detected_object in self.detected_objects:
            if detected_object.meas is not None:
                detected_object.meas.estimate_gps_location(self.latitude, self.longitude,
                                                           self.heading)

        logger.debug(f"Processing {len(self.detected_objects)} objects")
        logger.debug(self.detected_objects)

    def generate_vizualization_for_latest_data(self):
        vizualization = plot_object_with_distance(self.latest_data_tensor.view_img.copy(),
                                                  self.detected_objects, self.latest_data_tensor.n_xyz.copy())
        self.latest_data_tensor.visualized_data = vizualization
        return vizualization

    def get_latest_outputs(self, response_items: Dict):
        self.generate_vizualization_for_latest_data()

        for detected_object in self.detected_objects:
            detected_object.meas.estimate_gps_location(self.latitude, self.longitude,
                                                       self.heading)

        response = generate_response_dict(self.latest_data_tensor, self.detected_objects,
                                          response_items)

        return response


if __name__ == "__main__":
    import cv2
    # from src.offline_utils.frame_source import FrameSource

    # data_path = "/sdb-disk/vyzai/data/pxi_source/2024.12.02_Cityscape/2024-12-02_17-22-12/pxi/"
    # data_source = FrameSource(data_path, 0, 4)

    im_path = "/sdb-disk/vyzai/data/image_source/sykhiv/IMG_6947.jpeg"
    im_path = "/sdb-disk/vyzai/datasets/drones_detection/drone_dataset_yolo/dataset_txt/scene00676.jpg"
    data_tensor = DataPixelTensor.from_img_file(im_path)
    data_source = [data_tensor] * 20
    yml_path = resource_path('configs/project_la/manager.yaml')
    manager = ProjectLAManager(yml_path, device="cpu")

    video_writer = cv2.VideoWriter("output/project_la_test.mp4",
                                   cv2.VideoWriter.fourcc(*'mp4v'), 4,
                                   (2448, 2048), True)

    response_items = {
        "object_names": ["person", "drone"],
        "location": True,
        "images": {
            "visualized_data": False,
            "n_z": False,
            "n_xy": False,
            "n_xyz": False,
            "raw": False,
        }}

    for i, data_tensor in enumerate(data_source):
        manager.process(data_tensor)
        video_writer.write(manager.latest_data_tensor.visualized_data)
        cv2.imwrite("viz.png", manager.latest_data_tensor.visualized_data)

    video_writer.release()
