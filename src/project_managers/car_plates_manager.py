import os
import time
from datetime import datetime
import omegaconf
from ultralytics import YOLO
import cv2
import json
from paddleocr import PaddleOCR

from src.cv_module.cars.car_distance import CarRangeEstimation
from src.cv_module.cars.car_tracker import CarTracker

from src.cv_module.visualization import ResultsAnnotator

from src.cv_module.detectors.yolo_detector import YoloDetector


class CarPlatesOCRProcessingManager:
    """
    The main class of the current 'People Track' pipeline

    Contains a number of submodules
    """

    def __init__(self, crop_fov=False, obstacles_off=False,
                 roi=[0, 0, 0, 0], save_dir=None,
                 dist_refs=[0, 0]):
        self.obstacles_off = obstacles_off
        self.distance_scale_factor = 1.0
        self.measurements_people = {}

        self.crop_fov = crop_fov
        self.H, self.W = 2048, 2448

        self.roi = roi
        self.dist_refs = dist_refs

        config_dir = 'configs/ocr'
        model_dir = 'models/ocr'
        self.save_dir = save_dir

        self.config = omegaconf.OmegaConf.load(os.path.join(config_dir,
                                                            'car_plates_ocr_manager.yaml'))
        self.focal_length_mm = self.config.focal_length_mm
        self.focal_length_px = self.focal_length_mm / self.config.sensor_ratio

        self.plate_detector = YOLO(os.path.join(model_dir,
                                                'plate_detector.pt'))

        self.ocr_reader = PaddleOCR(lang='en', use_gpu=False)

        # Create all modules
        car_names = ["car", "bus", "truck"]
        model_config_dir = os.path.join(config_dir, "yolo_detector.yaml")
        self.dynamic_detector_cars = YoloDetector(model_config_dir, model_dir,
                                                  device=self.config.device,
                                                  processed_objects=car_names)

        self.car_tracker = CarTracker()
        self.car_range_estimator = CarRangeEstimation(self.focal_length_px,
                                                      (self.H, self.W))

        self.annotator = ResultsAnnotator(roi=self.roi)

        self.static_obstacles = []

        self.DEFAULT_HEADER = dict(version=1,
                                   dataType=1,
                                   width=2448,
                                   height=2048,
                                   bands=1,
                                   isScaled=False,
                                   isColor=False,
                                   numberOfMetadata=0,
                                   isNonBayer=False)

    def update_focal_length(self, value):
        self.focal_length_mm = value if value > 1 and value < 10000 else self.config.focal_length_mm
        self.focal_length_px = self.focal_length_mm / self.config.sensor_ratio

    def has_intersection(self, bbox1, bbox2):
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Перевірка перетину
        if x1_1 < x2_2 and x2_1 > x1_2 and y1_1 < y2_2 and y2_1 > y1_2:
            return True
        return False

    def process(self, data_tensor, frame_idx):
        tms = {}
        # frame, nz, nxy, nxyz, timestamp = frame
        frame = data_tensor["view_img"]

        # Process cars and plates
        car_objects, tms = self._detect_and_track_cars(frame,
                                                       data_tensor.created_at,
                                                       tms)
        car_objects = self._detect_plates(frame, car_objects, tms)
        car_objects = self._process_ocr(frame, car_objects, tms)

        # Process ranges and save results
        self.car_range_estimator.process(frame, car_objects)
        if self.save_dir is not None:
            self._save_car_data(frame, car_objects)

        # Visualize results
        annotated_frame = self._annotate_frame(frame, frame_idx, car_objects,
                                               data_tensor.n_xy, data_tensor.n_xyz,
                                               tms)

        return annotated_frame, tms

    def _detect_and_track_cars(self, frame, timestamp, tms):
        """Detect and track cars in the frame."""
        t1 = time.time()
        detected_cars = self.dynamic_detector_cars.predict(frame)
        car_objects = self.car_tracker.track(detected_cars)

        for car in car_objects:
            car.timestamp = timestamp

        tms["dynamic_cars"] = time.time() - t1
        return car_objects, tms

    def _detect_plates(self, frame, car_objects, tms):
        """Detect license plates for tracked cars."""
        t1 = time.time()
        for car in car_objects:
            if self.has_intersection(car.bbox, self.roi):
                x1, y1, x2, y2 = car.bbox
                plate_output = self.plate_detector(
                    frame[y1:y2, x1:x2], 
                    device=self.config.device, 
                    conf=0.2
                )[0]

                plate_bbox = plate_output.boxes.xyxy.detach().cpu().numpy().astype(int)

                if plate_bbox.size > 0 and plate_bbox.size <= 4:
                    xp1, yp1, xp2, yp2 = plate_bbox[0]
                    car.plate_bbox = [xp1 + x1, yp1 + y1, xp2 + x1, yp2 + y1]

        tms["plate_detect"] = time.time() - t1
        return car_objects

    def _process_ocr(self, frame, car_objects, tms):
        """Process OCR for detected license plates."""
        t1 = time.time()
        for car in car_objects:
            if car.plate_bbox is None:
                continue

            x1, y1, x2, y2 = car.plate_bbox
            if not self._is_plate_in_roi(x1, y1, x2, y2):
                continue

            ocr_frame = self._prepare_ocr_frame(frame[y1:y2, x1:x2])
            ocr_detections = self.ocr_reader.ocr(ocr_frame, det=False,
                                                 cls=False)
            car.ocr = ocr_detections

        tms["ocr"] = time.time() - t1
        return car_objects

    def _prepare_ocr_frame(self, frame):
        """Prepare frame for OCR processing."""
        ocr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(ocr_frame)

    def _is_plate_in_roi(self, x1, y1, x2, y2):
        """Check if the plate is within the ROI."""
        return (y1 > self.roi[1] and y2 < self.roi[3] and 
                x1 > self.roi[0] and x2 < self.roi[2])

    def _save_car_data(self, frame, car_objects):
        """Save car data and images."""
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for car in car_objects:
            if not self.has_intersection(car.bbox, self.roi):
                continue

            data = self._prepare_car_data(car, current_time)
            self._save_car_images(frame, car, current_time)
            self._save_car_info(car, data, current_time)

    def _prepare_car_data(self, car, current_time):
        """Prepare car data for saving."""
        ocr_data = None
        car_plate = None

        if car.ocr is not None:
            ocr_data = [
                {"plate": item[0], "confidence": item[1]}
                for item in car.ocr[0]
            ]
            car_plate = car.plate

        data = {
            'name': f'{car.id}_car_{current_time}.jpg',
            'track_id': int(car.id),
            'type': car.detection.name,
            'velocity': car.velocity,
            'plate_bbox': None if car_plate is None else [int(coord) for coord in car_plate],
            'ocr': ocr_data
        }

        return data

    def _save_car_images(self, frame, car, current_time):
        """Save car and plate images."""
        car_image_path = (self.save_dir /
                          f'images/{car.id}_{car.detection.name.lower()}_{current_time}.jpg')
        car_crop = frame[car.bbox[1]:car.bbox[3], car.bbox[0]:car.bbox[2]]
        cv2.imwrite(str(car_image_path), car_crop)

    def _save_car_info(self, car, data, current_time):
        """Save car information to YAML file."""
        info_path = (self.save_dir / f'info/{car.id}_{car.detection.name.lower()}_{current_time}.yaml')
        if not info_path.parent.exists():
            info_path.parent.mkdir(parents=True)

        with open(info_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    def _annotate_frame(self, frame, frame_idx, car_objects, nxy, nxyz, tms):
        """Annotate frame with detection results."""
        t1 = time.time()
        annotated_frame = self.annotator.process(
            frame, frame_idx, [], car_objects, [], nxy, nxyz, self.crop_fov
        )
        tms["visual"] = time.time() - t1
        return annotated_frame
