from omegaconf import OmegaConf
from typing import Dict, List
from loguru import logger

from src.data_pixel_tensor import DataPixelTensor
from src.project_managers.outputs.project_la_outputs import generate_response_dict
from src.cv_module.distance_measurers.width_measurer import WidthDistanceMeasurer
from src.cv_module.visualization import plot_object_with_distance
from src.cv_module.basic_object import BasicObjectWithDistance
from src.drone_pipeline.detector_yolov8 import YoloV8Detector
from src.drone_pipeline.segmenter import HybridSegmenter, SamBBoxSegmenter, SamConfig, YoloV8SegFallback
from src.drone_pipeline.classifier_silhouette import SilhouetteClassifier
from src.drone_pipeline.tracker_csrt import CSRTTracker
from src.drone_pipeline.pipeline import DronePipelineManager
from src.utils.common import resource_path


class ProjectDronesManager:
    def __init__(self, config_path: str, device: str = "cuda"):
        self.config = OmegaConf.load(resource_path(config_path))

        self.imsize = (self.config.im_height, self.config.im_width)
        self.focal_length_mm = self.config.focal_length_mm
        self.focal_length_px = self.focal_length_mm / self.config.sensor_ratio

        # ------------------------------------------------------------------
        # Modular drone pipeline (detection + segmentation + classification)
        # ------------------------------------------------------------------
        pipeline_cfg_path = resource_path(self.config.drone_pipeline_config)
        self.drone_pipeline_cfg = OmegaConf.load(pipeline_cfg_path)

        # Detector (YOLOv8 / YOLOv11x)
        det_cfg = self.drone_pipeline_cfg.detector
        labels_cfg = det_cfg.get("labels", ["drone"])
        processed_labels = None if labels_cfg is None else list(labels_cfg)
        models_dir = det_cfg.models_dir or ""
        self.drones_detector = YoloV8Detector(
            config_path=resource_path(det_cfg.config_path),
            models_dir=resource_path(models_dir) if models_dir else "",
            device=det_cfg.get("device", device),
            processed_labels=processed_labels,
        )

        # Segmentation backends
        seg_cfg = self.drone_pipeline_cfg.segmentation
        sam_segmenter = None
        if getattr(seg_cfg.sam, "enabled", False):
            sam_conf = SamConfig(
                checkpoint=resource_path(seg_cfg.sam.checkpoint),
                model_type=seg_cfg.sam.get("model_type", "vit_l"),
                device=seg_cfg.sam.get("device", device),
            )
            try:
                sam_segmenter = SamBBoxSegmenter(sam_conf)
            except ImportError as e:
                logger.warning(f"SAM not available, will rely on YOLOv8-Seg fallback only: {e}")

        yolo_fallback = None
        if getattr(seg_cfg.yolo_fallback, "enabled", False):
            yolo_fallback = YoloV8SegFallback(
                model_path=resource_path(seg_cfg.yolo_fallback.model_path),
                device=seg_cfg.yolo_fallback.get("device", device),
            )

        self.drone_segmenter = HybridSegmenter(sam=sam_segmenter, yolo_fallback=yolo_fallback)

        # Classification (silhouette-based)
        cls_cfg = self.drone_pipeline_cfg.classification
        self.drone_classifier = SilhouetteClassifier(
            silhouettes_dir=cls_cfg.silhouettes_dir,
            image_size=int(cls_cfg.image_size),
            top_k=int(cls_cfg.top_k),
            chamfer_weight=float(cls_cfg.chamfer_weight),
        )

        # Tracking: CSRT-based image tracker (YOLO детектить рідко, CSRT трекає кожен кадр)
        tracker = CSRTTracker()

        # Orchestrator
        sched_cfg = self.drone_pipeline_cfg.scheduler
        detection_interval = int(sched_cfg.detection_interval)
        detection_interval_tracked = int(getattr(sched_cfg, "detection_interval_tracked", detection_interval))
        recovery_interval = int(getattr(sched_cfg, "recovery_interval", detection_interval))
        stale_detection_frames = int(getattr(sched_cfg, "stale_detection_frames", 15))
        self.drone_pipeline = DronePipelineManager(
            detector=self.drones_detector,
            segmenter=self.drone_segmenter,
            classifier=self.drone_classifier,
            tracker=tracker,
            detection_interval=detection_interval,
            detection_interval_tracked=detection_interval_tracked,
            max_workers=int(sched_cfg.max_workers),
            recovery_interval=recovery_interval,
            stale_detection_frames=stale_detection_frames,
        )

        # Distance measurer based on physical drone width
        self.drones_distance_measurer = WidthDistanceMeasurer(
            self.focal_length_px,
            self.imsize,
            base_width_in_meters=self.drones_detector.get_drone_width(),
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

        # Update distance measurer to use the new focal length
        self.drones_distance_measurer = WidthDistanceMeasurer(
            self.focal_length_px,
            self.imsize,
            base_width_in_meters=self.drones_detector.get_drone_width(),
        )

    def _tracks_to_basic_objects(self, tracks) -> List[BasicObjectWithDistance]:
        """Convert tracked states to BasicObjectWithDistance for downstream code."""
        basic_objects: List[BasicObjectWithDistance] = []
        for tr in tracks:
            if tr.label != "drone":
                continue
            obj = BasicObjectWithDistance(
                id=tr.track_id,
                bbox=tr.bbox,
                conf=tr.score,
                mask=tr.mask,
                name=tr.label,
                timestamp=tr.timestamp,
            )
            if tr.classification is not None:
                obj.classification = tr.classification  # type: ignore[attr-defined]
            basic_objects.append(obj)
        return basic_objects

    def process(self, data_tensor: DataPixelTensor):
        frame = data_tensor["view_img"]
        timestamp = float(getattr(data_tensor, "created_at", 0.0))

        tracks = self.drone_pipeline.update(frame, timestamp)
        detected_drones = self._tracks_to_basic_objects(tracks)

        if detected_drones:
            measurements = self.drones_distance_measurer.process(detected_drones)
            for drone in detected_drones:
                drone.meas = measurements[drone.id]

        detected_people: List[BasicObjectWithDistance] = []
        detected_cars: List[BasicObjectWithDistance] = []

        self.detected_objects = detected_people + detected_cars + detected_drones
        self.latest_data_tensor = data_tensor

        for detected_object in self.detected_objects:
            if detected_object.meas is not None:
                detected_object.meas.estimate_gps_location(
                    self.latitude,
                    self.longitude,
                    self.heading,
                )

        logger.debug(f"Processing {len(self.detected_objects)} objects")
        logger.debug(self.detected_objects)

    def generate_vizualization_for_latest_data(self):
        vizualization = plot_object_with_distance(
            self.latest_data_tensor.view_img.copy(),
            self.detected_objects,
            self.latest_data_tensor.n_xyz.copy(),
        )
        self.latest_data_tensor.visualized_data = vizualization
        return vizualization

    def get_latest_outputs(self, response_items: Dict):
        self.generate_vizualization_for_latest_data()

        for detected_object in self.detected_objects:
            detected_object.meas.estimate_gps_location(
                self.latitude,
                self.longitude,
                self.heading,
            )

        response = generate_response_dict(
            self.latest_data_tensor,
            self.detected_objects,
            response_items,
        )

        return response
