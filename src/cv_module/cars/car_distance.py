import numpy as np
from typing import Tuple, List
# import os
# import timm
import torch
import cv2

from .car import Car
from src.cv_module.distance_measurers.height_measurer import HeightDistanceMeasurer

MEDIUM_VEHICLE_LENGTH = 4.370
MEDIUM_VEHICLE_HEIGHT = 1.458
MEDIUM_VEHICLE_WIDTH = 1.883


class CarRangeEstimation:
    def __init__(self, focal_length: int,
                 im_size: Tuple[int, int]):
        self._focal_length = focal_length
        self.im_size = im_size

        self.bbox_measurer = HeightDistanceMeasurer(focal_length, im_size,
                                                    MEDIUM_VEHICLE_HEIGHT)

        # self._classification_model = timm.create_model("efficientnet_b0",
        #                                                checkpoint_path=os.path.join(model_dir, "car_classification.pt"),
        #                                                num_classes=3).eval()

        self._class_name_to_sizes = {
            "small": {
                "length": 0,
                "width": 0,
                "height": 1.4
            },
            "medium": {
                "length": MEDIUM_VEHICLE_LENGTH,
                "width": MEDIUM_VEHICLE_WIDTH,
                "height": MEDIUM_VEHICLE_HEIGHT
            },
            "large": {
                "length": 0,
                "width": 0,
                "height": 2.4
            },
        }

        self._classification_model_labels = {
            0: "small",
            1: "medium",
            2: "large"
        }

    def preprocess_classification_input(self, image: np.ndarray):
        image = image.copy()

        height, width = image.shape[:2]

        if height != width:
            if height > width:
                added_width = height - width
                left_added = added_width // 2
                right_added = added_width - left_added
                top_added = 0
                bottom_added = 0
            elif width > height:
                added_height = width - height
                top_added = added_height // 2
                bottom_added = added_height - top_added
                left_added = 0
                right_added = 0

            image = cv2.copyMakeBorder(image, top_added, bottom_added,
                                       left_added, right_added,
                                       borderType=cv2.BORDER_CONSTANT, value=0)

        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)

        image = np.float32(image / 255)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        image -= mean
        image /= std

        image_tensor = torch.tensor(image).permute((2, 0, 1))

        return image_tensor

    def process_one(self, frame, car: Car):
        # x1, y1, x2, y2 = car.bbox

        # crop = frame[y1:y2, x1:x2]
        # classification_input = self.preprocess_classification_input(crop)

        # cls_model_output = self._classification_model(classification_input.unsqueeze(0))[0]
        # class_idx = cls_model_output.argmax(0).item()
        # class_name = self._classification_model_labels[class_idx]

        class_name = "medium"
        height_in_meters = self._class_name_to_sizes[class_name]["height"]
        car.height = height_in_meters

        car.set_current_type(class_name)

        meas = self.bbox_measurer.process_one(car)

        return meas

    def process(self, frame: np.ndarray, cars: List["Car"]):
        for car in cars:
            car_measurement = self.process_one(frame, car)

            car.set_measurement(car_measurement)

        return cars
