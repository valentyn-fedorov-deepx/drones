import os
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from omegaconf import OmegaConf
import numpy as np
import cv2
from loguru import logger
import torch
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional, Any
from src.cv_module.detected_object import DetectedObject
from src.utils.common import resource_path


class YoloDetector:
    def __init__(self, config_path: str, models_dir: str, device: str,
                 processed_objects: List[str] = ['person'],
                 name_map: Optional[Dict] = None, refine_masks: bool = False):
        self.device = device
        self.config = OmegaConf.load(config_path)
        self.drone_width = self.config.drone_width_meters
        model_path = os.path.join(models_dir, self.config.model_name)
        model_path = resource_path(model_path)
        self._model_path = model_path
        self._is_pytorch = str(model_path).endswith(".pt")
        self.model = YOLO(model_path)
        # Only PyTorch models support .to(device); exported formats manage device internally
        try:
            if self._is_pytorch:
                self.model = self.model.to(device)
        except Exception as e:
            logger.warning(f"Failed to move model to {device}: {e}")
        self._processed_objects = processed_objects
        self._name_map = name_map
        refine_masks = False
        self._refine_masks = refine_masks
        if refine_masks:
            import segmentation_refinement as refine
            self.refiner = refine.Refiner(device='cuda:0')

    def get_drone_width(self) -> float:
        """Return preconfigured drone width in meters."""
        return self.drone_width

    def predict_cropped(self, image: np.ndarray,
                        center_point: Tuple[int, int] = (1024, 1224),
                        crop_size: Tuple[int, int] = (640, 640)):
        x_center, y_center = center_point
        crop_h, crop_w = crop_size

        x1_crop = x_center - (crop_w // 2)
        y1_crop = y_center - (crop_h // 2)
        x2_crop = x1_crop + crop_w
        y2_crop = y1_crop + crop_h

        image_cropped = image[y1_crop:y2_crop, x1_crop:x2_crop]
        predicted_objects = self.predict(image_cropped)
        for predicted_object in predicted_objects:
            x1, y1, x2, y2 = predicted_object.bbox
            crop_adjusted_bbox = np.array((x1 + x1_crop, y1 + y1_crop,
                                           x2 + x1_crop, y2 + y1_crop),
                                          dtype=int)

            predicted_object.bbox = crop_adjusted_bbox

        return predicted_objects

    def predict(self, image: np.ndarray) -> List[DetectedObject]:
        import time
        start_time = time.time()

        results = self.model(image, imgsz=self.config.infer_imgsize,
                             conf=self.config.model_conf, verbose=False)[0]
        logger.debug(f"Yolo inference time: {time.time() - start_time:.3f} sec")

        # cv2.imwrite("debug_yolo.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # import ipdb; ipdb.set_trace()

        detected_objects = list()

        import time
        start_time = time.time()

        names_inv = {name: idx for idx, name in results.names.items()}
        if self._processed_objects is None:
            detected_objects_idxs = range(results.boxes.cls.shape[0])
        else:
            selected_class_idx = [names_inv[class_name] for class_name in self._processed_objects]

            detected_objects_idxs = np.where((np.isin(results.boxes.cls.detach().cpu().numpy().astype(int),
                                                      selected_class_idx)))[0]
        logger.debug(f"Yolo filtering time: {time.time() - start_time:.3f} sec")

        for idx in detected_objects_idxs:
            start_time = time.time()
            box = results.boxes.xyxy[idx].detach().cpu().numpy().astype(int)
            conf = results.boxes.conf[idx].item()
            class_idx = results.boxes.cls[idx].item()
            class_name = results.names[class_idx]
            if self._name_map:
                class_name = self._name_map.get(class_name, class_name)

            if results.masks is None:
                final_mask = None
            else:
                original_mask = results.masks.data[idx].detach().cpu().numpy()
                mask_h, mask_w = original_mask.shape
                x1_n, y1_n, x2_n, y2_n = results.boxes.xyxyn[idx].detach().cpu().numpy()
                x1_mask = int(x1_n * mask_w)
                y1_mask = int(y1_n * mask_h)
                x2_mask = int(x2_n * mask_w)
                y2_mask = int(y2_n * mask_h)

                cropped_mask = original_mask[y1_mask:y2_mask, x1_mask:x2_mask].astype(np.uint8)

                box_h = box[-1] - box[1]
                box_w = box[-2] - box[0]

                resized_mask = cv2.resize(cropped_mask, (box_w, box_h),
                                          interpolation=cv2.INTER_NEAREST)

                # cv2.imwrite('debug_yolo_box.png', cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR))
                # cv2.imwrite('debug_yolo_box_mask.png', resized_mask * 255)

                final_mask = resized_mask
                if self._refine_masks:
                    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    full_mask[box[1]:box[3], box[0]:box[2]] = resized_mask

                    # cv2.imwrite('debug_yolo_box_mask_full.png', full_mask * 255)
                    refined_mask = self.refiner.refine(image, full_mask * 255, fast=False, L=400)
                    # cv2.imwrite('debug_yolo_box_mask_refined.png', refined_mask)
                    final_mask = np.round(refined_mask[box[1]:box[3], box[0]:box[2]] / 255)
                    final_mask = final_mask.astype(np.uint8)
                # cv2.imwrite('debug_yolo_box_mask_final.png', final_mask * 255)

            # import ipdb; ipdb.set_trace()

            logger.debug(f"Yolo mask processing time: {time.time() - start_time:.3f} sec")

            import time
            start_time = time.time()
            detected_object = DetectedObject(mask=final_mask, name=class_name,
                                             bbox=box, conf=conf,
                                             imsize=image.shape[:2])
            logger.debug(f"DetectedObject creation time: {time.time() - start_time:.3f} sec")
            detected_objects.append(detected_object)

        return detected_objects

class YoloDetectorTiled(YoloDetector):
    """
    Works same as YoloDetector, but no downsampling performed.
    Input image divided by tiles and processed in thread pool instead.
    """
    def __init__(self, config_path: str, models_dir: str, device: str,
                 processed_objects: List[str] = ['person'],
                 name_map: Optional[Dict] = None, refine_masks: bool = False):
        super().__init__(
            config_path, models_dir, device,
            processed_objects, name_map, refine_masks
        )
        # Preload N copies of the model (one per worker)
        self.num_workers = self.config.model_num_workers
        if getattr(self, "_is_pytorch", False):
            self.models = [deepcopy(self.model).to(device) for _ in range(self.num_workers)]
        else:
            logger.warning(
                "Non-PyTorch model detected; disabling model copies and forcing num_workers=1"
            )
            self.num_workers = 1
            self.models = [self.model]

        # Tile parameters
        self.tile_size = self.config.tile_size
        self.stride = self.tile_size - self.config.tile_overlap
        self.nms_iou_threshold = self.config.non_max_suppression_intersection_over_union_threshold

        # Model parameters
        self.model_conf_threshold = self.config.model_conf
        self.model_do_augment = self.config.model_augment

    def _prepare_tiles(self, image: np.ndarray) -> tuple[list[int], list[int]]:
        """Compute tile coordinates dynamically based on image size."""
        h, w = image.shape[:2]
        x_steps = list(range(0, w, self.stride))
        y_steps = list(range(0, h, self.stride))

        if x_steps[-1] + self.tile_size < w:
            x_steps.append(w - self.tile_size)
        if y_steps[-1] + self.tile_size < h:
            y_steps.append(h - self.tile_size)

        return x_steps, y_steps

    def _prepare_tile_args(self, image: np.ndarray) -> list[
        Tuple[
            int,  # worker_id
            np.ndarray,  # tile
            int,  # x offset
            int,  # y offset
            tuple[int, int],  # image shape
            float,  # confidence threshold
            bool  # do_augment
        ]
    ]:
        """Make args for thread workers."""
        x_steps, y_steps = self._prepare_tiles(image)
        tiles_args = []

        tile_id = 0
        for y in y_steps:
            for x in x_steps:
                tile = image[y:y + self.tile_size, x:x + self.tile_size]

                # Pad smaller tiles
                th, tw = tile.shape[:2]
                pad_b = self.tile_size - th
                pad_r = self.tile_size - tw
                if pad_b > 0 or pad_r > 0:
                    tile = cv2.copyMakeBorder(
                        tile, 0, pad_b, 0, pad_r,
                        cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )

                worker_id = tile_id % self.num_workers
                tiles_args.append((
                    worker_id, tile, x, y, image.shape[:2],
                    self.model_conf_threshold, self.model_do_augment
                ))
                tile_id += 1

        return tiles_args

    def _infer_tile(
            self, worker_id: int, tile: np.ndarray,
            x: int, y: int, image_shape: Tuple[int, int],
            conf: float, augment=False
    )-> Results:
        """Perform model's inference for given tile."""
        # Each worker uses its own model
        model = self.models[worker_id]
        results_tile = model(tile, conf=conf, verbose=False, augment=augment)[0]

        # Offset boxes
        if results_tile.boxes.data.shape[0] > 0:
            boxes = results_tile.boxes.data.clone()
            boxes[:, [0, 2]] += x
            boxes[:, [1, 3]] += y
            results_tile.boxes = Boxes(boxes, orig_shape=image_shape)

        return results_tile

    def _fuse_tiles_results(
            self, image: np.ndarray, tiles_results: List[Results]
    ) -> Results:
        """Merge list into one Results."""
        if not tiles_results:
            return Results(image, self.models[0].model, [])

        combined = deepcopy(tiles_results[0])
        all_boxes = [r.boxes.data for r in tiles_results if r.boxes.data.shape[0] > 0]

        # Merge bounding boxes
        if all_boxes:
            merged = torch.cat(all_boxes, dim=0)
            combined.boxes = Boxes(merged, orig_shape=image.shape[:2])
        else:
            combined.boxes = Boxes(torch.zeros((0, 6)), orig_shape=image.shape[:2])

        if not self._refine_masks:
            return combined

        # Merge masks
        if combined.masks is not None:
            all_masks = [r.masks.data for r in tiles_results if r.masks is not None]
            if all_masks:
                combined.masks.data = torch.cat(all_masks, dim=0)

        return combined

    def _run_yolo_tiled(self, image: np.ndarray) -> Results:
        """Parallel YOLO tile inference."""
        tiles_args = self._prepare_tile_args(image)
        tiles_results = []

        # Run in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for r in executor.map(lambda p: self._infer_tile(*p), tiles_args):
                tiles_results.append(r)

        return self._fuse_tiles_results(image, tiles_results)

    def _process_results(
            self, image: np.ndarray, results: Results
    ) -> Tuple[
        List[List[float]],                    # dets
        Optional[List[Optional[np.ndarray]]]  # masks
    ]:
        """Build detections and masks lists from YOLO results."""
        dets = []
        masks = None  # aligned list: each entry either full-image mask (H,W) or None
        for idx in range(results.boxes.data.shape[0]):
            box = results.boxes.xyxy[idx].detach().cpu().numpy().astype(int)
            conf = results.boxes.conf[idx].item()
            cls = int(results.boxes.cls[idx].item())

            x1, y1, x2, y2 = box.tolist()
            dets.append([x1, y1, x2, y2, conf, cls])

            # Build full-image mask if available
            if results.masks is not None:
                masks = []
                original_mask = results.masks.data[idx].detach().cpu().numpy()
                # results.boxes.xyxyn are normalized to orig_shape of masked tile; but after tiling we merged masks as full-image masks in earlier pipeline.
                # For safety, create a full-image mask placeholder filled by resized mask into bbox.
                try:
                    box_h = y2 - y1
                    box_w = x2 - x1
                    cropped_mask = original_mask
                    mask_full = np.zeros(image.shape[:2], dtype=np.uint8)
                    resized_mask = cv2.resize(cropped_mask, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
                    mask_full[y1:y2, x1:x2] = (resized_mask > 0).astype(np.uint8)
                except Exception:
                    mask_full = None
                masks.append(mask_full)

        return dets, masks

    def _convert_results(
            self, image: np.ndarray,
            detections: List[List[float]],
            masks: Optional[List[Optional[np.ndarray]]],
            timestamp: Optional[float] = None
    ) -> List[DetectedObject]:
        """Convert final detections to DetectedObject list."""
        detected_objects = []

        # Traverse detections
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls_idx = det
            class_name = self.models[0].names[int(cls_idx)] if hasattr(self.models[0], "names") else str(int(cls_idx))
            if self._processed_objects is not None:
                if class_name not in self._processed_objects:
                    continue
            if self._name_map:
                class_name = self._name_map.get(class_name, class_name)

            # Process the mask if available
            mask_full = None
            if masks is not None and i < len(masks):
                mf = masks[i]
                if mf is not None:
                    # convert full-image mask to bbox-local mask to store in DetectedObject
                    bx1, by1, bx2, by2 = int(x1), int(y1), int(x2), int(y2)
                    try:
                        m_crop = mf[by1:by2, bx1:bx2]
                        box_h = by2 - by1
                        box_w = bx2 - bx1
                        if m_crop.shape != (box_h, box_w):
                            m_crop = cv2.resize(m_crop, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
                        mask_full = (m_crop > 0).astype(np.uint8)
                    except Exception:
                        mask_full = None

            detected_objects.append(
                DetectedObject(
                    mask=mask_full, name=class_name,
                    bbox=np.array([int(x1), int(y1), int(x2), int(y2)]),
                    conf=float(conf), imsize=image.shape[:2],
                    timestamp=timestamp
                )
            )

        return detected_objects

    def _unpack_detections(
            self, dets: List[List[float]]
    )-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert list of detections to numpy arrays (one per each parameter)."""
        arr = np.array(dets)
        x1 = arr[:, 0]
        y1 = arr[:, 1]
        x2 = arr[:, 2]
        y2 = arr[:, 3]
        scores = arr[:, 4]
        classes = arr[:, 5]
        return x1, y1, x2, y2, scores, classes

    def _get_unsuppressed_idx(
            self, x1: np.ndarray, y1: np.ndarray,
            x2: np.ndarray, y2: np.ndarray,
            scores: np.ndarray, classes: np.ndarray
    ) -> List[int]:
        """Perform non-max suppression, return list of indexes of detections to keep."""
        keep_indices = []

        # Traverse object classes
        for cls in np.unique(classes):
            idxs = np.where(classes == cls)[0]
            cls_scores = scores[idxs]
            cls_idxs = idxs[np.argsort(-cls_scores)]

            # Process detections starting from highest confidence
            while len(cls_idxs) > 0:
                i = cls_idxs[0]
                keep_indices.append(int(i))

                if len(cls_idxs) == 1:
                    break

                rest = cls_idxs[1:]
                xx1 = np.maximum(x1[i], x1[rest])
                yy1 = np.maximum(y1[i], y1[rest])
                xx2 = np.minimum(x2[i], x2[rest])
                yy2 = np.minimum(y2[i], y2[rest])

                # Calculate Intersection Over Union
                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                inter = w * h

                area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
                area_rest = (x2[rest] - x1[rest]) * (y2[rest] - y1[rest])
                union = area_i + area_rest - inter
                iou = inter / (union + 1e-6)

                # Drop all intersected detections with lower confidence
                cls_idxs = cls_idxs[1:][iou < self.nms_iou_threshold]

        return keep_indices

    def _filter_detections(
            self, dets: List[List[float]],
            keep_indices: List[int],
            masks: Optional[List[Optional[np.ndarray]]] = None,
    ) -> Tuple[List[List[float]], Optional[List[Optional[np.ndarray]]]]:
        """Filter detections and masks by given indexes."""
        # Preserve original order by indexes appear in keep_indices
        kept = [dets[i] for i in keep_indices]
        kept_masks = None
        if masks is not None:
            kept_masks = [masks[i] for i in keep_indices]
        return kept, kept_masks

    def non_max_suppression(
            self, dets: List[List[float]],
            masks: Optional[List[Optional[np.ndarray]]] = None
    ) -> Tuple[List[List[float]], Optional[List[Optional[np.ndarray]]]]:
        """
        Apply non-max suppression, return list of kept
        detections and list of their masks (same order).
        """
        if len(dets) == 0:
            return [], []

        x1, y1, x2, y2, scores, classes = self._unpack_detections(dets)
        keep_indices = self._get_unsuppressed_idx(x1, y1, x2, y2, scores, classes)
        return self._filter_detections(dets, keep_indices, masks)

    def predict(self, image: np.ndarray) -> List[DetectedObject]:
        """Perform Yolo inference on image from camera."""
        import time
        start_time = time.time()

        # Run YOLO tiled
        results = self._run_yolo_tiled(image)
        logger.debug(f"Yolo inference time: {time.time() - start_time:.3f} sec")

        # Build dets + masks lists from YOLO results
        dets, masks = self._process_results(image, results)

        # Run NMS on combined list (nms returns kept dets and kept masks)
        dets_nms, masks_nms = self.non_max_suppression(
            dets,
            masks=masks
        )

        # Convert final dets to DetectedObject list (masks taken from masks_nms per det)
        return self._convert_results(image, dets_nms, masks_nms)


class YoloDetectorTiledContexted(YoloDetectorTiled):
    """
    On top of YoloDetectorTiled, it calculates surrounding average gradient
    for each inference and correct confidence according to it.
    For local gradients lower, then general mean - confidence decreased,
    meanwhile for gradients higher, then general mean - increased.
    """
    def __init__(self, config_path: str, models_dir: str, device: str,
                 processed_objects: List[str] = ['person'],
                 name_map: Optional[Dict] = None, refine_masks: bool = False):
        super().__init__(
            config_path, models_dir, device,
            processed_objects, name_map, refine_masks
        )
        # Parameters to calculate surrounding mean gradient
        self.bg_percentile = self.config.bg_gradient_general_drop_percentile
        self.bg_bbox_size_factor = self.config.bg_gradient_bbox_expand_fraction

        # Parameters to correct confidence by target surrounding mean gradient
        self.bg_local_min = self.config.bg_gradient_lower_bound_fraction_of_mean
        self.bg_local_max =self.config.bg_gradient_higher_bound_fraction_of_mean
        # Confidence of target with 1 / bg_local_min fraction of
        # image mean gradient will be decreased on:
        self.bg_conf_max_decrease = self.config.bg_gradient_max_conf_decrease
        # Confidence of target with bg_local_max factor of
        # image mean gradient will be increased on:
        self.bg_conf_max_increase = self.config.bg_gradient_max_conf_increase
        self.conf_threshold = self.config.general_conf_threshold

    def _get_background_variance_score(
            self, image: np.ndarray
    ) -> Tuple[float, float]:
        """
        Computes a robust gradient magnitude map and a robust average gradient value.
        High percentile values are dropped to avoid influence of outlier pixels.

        Returns:
            grad_mag: full gradient magnitude map
            avg_grad: robust average gradient
        """
        gray = np.mean(image, axis=2).astype(np.float32)

        # Sobel gradients
        # TODO try more ksize (blurring?)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx * gx + gy * gy)

        # robust mean: drop highest percentile
        flat = grad_mag.flatten()
        cutoff = np.percentile(flat, self.bg_percentile)

        robust_pixels = flat[flat <= cutoff]
        avg_grad = robust_pixels.mean()

        return grad_mag, float(avg_grad)

    def _get_object_gradient(
            self,
            grad_magnitude: np.ndarray,
            obj: DetectedObject
    ) -> Tuple[Optional[float], Optional[DetectedObject]]:
        """
        Calculate surrounding average gradient for an object.
        If returned object is not None, that should be
        processed without surrounding gradient checking.
        """
        img_h, img_w = grad_magnitude.shape
        x1, y1, x2, y2 = obj.bbox
        conf = obj.conf

        # Object size and expansion
        w = x2 - x1
        h = y2 - y1
        expand = self.bg_bbox_size_factor * 0.5 * (w + h)

        # Inner (object) box
        xi1 = max(0, int(x1))
        yi1 = max(0, int(y1))
        xi2 = min(img_w - 1, int(x2))
        yi2 = min(img_h - 1, int(y2))

        # Outer (expanded box)
        xo1 = max(0, int(x1 - expand))
        yo1 = max(0, int(y1 - expand))
        xo2 = min(img_w - 1, int(x2 + expand))
        yo2 = min(img_h - 1, int(y2 + expand))

        # Extract regions
        inner = grad_magnitude[yi1:yi2, xi1:xi2]
        outer = grad_magnitude[yo1:yo2, xo1:xo2]

        # Surrounding = outer minus inner
        if inner.size == 0 or outer.size == 0:
            obj.conf = conf
            if obj.conf >= self.conf_threshold:
                return None, obj

        inner_mask = np.zeros_like(outer, dtype=bool)
        inner_mask[
            (yi1 - yo1):(yi2 - yo1),
            (xi1 - xo1):(xi2 - xo1)
        ] = True

        surround = outer[~inner_mask]
        if surround.size == 0:
            obj.conf = conf
            if obj.conf >= self.conf_threshold:
                return None, obj

        return float(surround.mean()), None

    def _adjust_object_confidence(
            self,
            obj: DetectedObject,
            obj_grad: float,
            avg_grad: float
    ) -> None:
        """
        Update object confidence according to its surrounding gradient.

        Rules:
          - obj_grad >= 2*avg_grad  => +max_increase
          - obj_grad <= 0.5*avg_grad => -max_decrease
          - Linear interpolation in between.
        """
        grad_low_bound = avg_grad / self.bg_local_min
        grad_high_bound = self.bg_local_max * avg_grad

        if obj_grad <= grad_low_bound:
            # strongest downscale
            new_conf = obj.conf * (1.0 - self.bg_conf_max_decrease)
        elif obj_grad >= grad_high_bound:
            # strongest upscale
            new_conf = obj.conf * (1.0 + self.bg_conf_max_increase)
        else:
            # linear interpolation between the two
            t = (obj_grad - grad_low_bound) / (grad_high_bound - grad_low_bound)
            # t = 0 → decrease, t = 1 → increase
            adj = (-self.bg_conf_max_decrease) * (1 - t) + self.bg_conf_max_increase * t
            new_conf = obj.conf * (1.0 + adj)

        obj.conf = float(np.clip(new_conf, 0.0, 1.0))

    def filter_by_background_variance(
            self,
            image: np.ndarray,
            detected_objects: List[DetectedObject]
    ) -> List[DetectedObject]:
        """Adjusted confidences then filter detections (based on surrounding gradient)"""
        grad_mag, avg_grad = self._get_background_variance_score(image)

        adjusted_objects = []
        for obj in detected_objects:
            obj_grad, checked_obj = self._get_object_gradient(grad_mag, obj)
            if checked_obj is not None:
                adjusted_objects.append(checked_obj)
                continue

            self._adjust_object_confidence(obj, obj_grad, avg_grad)
            if obj.conf >= self.conf_threshold:
                adjusted_objects.append(obj)

        return adjusted_objects

    def predict(self, image: np.ndarray) -> List[DetectedObject]:
        """
        Perform Yolo inference and correct confidences
        according to surrounding average gradients
        """
        return self.filter_by_background_variance(
            image,
            super().predict(image)
        )

class YoloDetectorTiledContextedTracked(YoloDetectorTiledContexted):
    """
    On top of YoloDetectorTiledContexted, it applies CSRT tracker on each inference,
    so that whenever the model can't recognize object on a new frame, tracker comes to play.
    """
    def __init__(self, config_path: str, models_dir: str, device: str,
                 processed_objects: List[str] = ['person'],
                 name_map: Optional[Dict] = None, refine_masks: bool = False):
        super().__init__(
            config_path, models_dir, device,
            processed_objects, name_map, refine_masks
        )
        # Tracker configuration
        self.trackers = {}             # tid -> cv2 tracker
        self.tracked_objects = {}      # tid -> {"cls_idx": int, "mask": np.ndarray (H,W), "bbox": (x1,y1,x2,y2)}
        self.next_track_id = 0
        self.edge_margin = self.config.tracker_frame_edge_drop_margin_pix
        self.tracker_nms_bbox_expansion = self.config.tracker_bbox_expand_fraction

        # Tools for synthetic masks
        # Store previous frame for optical flow
        self.prev_frame_gray = None
        # Optical flow object (fast preset)
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    def _get_object_gradient(
            self,
            grad_magnitude: np.ndarray,
            obj: DetectedObject
    ) -> Tuple[Optional[float], Optional[DetectedObject]]:
        """
        Calculate surrounding average gradient for an object.
        If returned object is not None, that should be
        processed without surrounding gradient checking.
        """
        # Exactly zero confidence has objects from the tracker (to be robustly suppressed by NMS).
        # But to bypass confidence threshold, need to set the lowest valid confidence instead
        if obj.conf == 0.0:
            obj.conf = self.conf_threshold
            return None, obj

        return super()._get_object_gradient(grad_magnitude, obj)

    def non_max_suppression(
            self, dets: List[List[float]],
            masks: Optional[List[Optional[np.ndarray]]] = None
    ) -> Tuple[List[List[float]], Optional[List[Optional[np.ndarray]]]]:
        """
        Apply non-max suppression, return list of kept
        detections and list of their masks (same order).
        """
        if len(dets) == 0:
            return [], []

        x1, y1, x2, y2, scores, classes = self._unpack_detections(dets)

        # Expand bounding boxes from tracker
        bw = (scores == 0.0) | (scores == self.conf_threshold)
        sizes = x2[bw]-x1[bw] + y2[bw]-y1[bw]
        expands = self.tracker_nms_bbox_expansion * 0.5 * sizes
        x1[bw] -= expands
        x2[bw] += expands
        y1[bw] -= expands
        y2[bw] += expands

        keep_indices = self._get_unsuppressed_idx(x1, y1, x2, y2, scores, classes)
        return self._filter_detections(dets, keep_indices, masks)

    def _create_tracker(self) -> Any:
        """Tracker should be recreated for each new image."""
        # TODO try to average few last images
        return cv2.legacy.TrackerCSRT_create()

    def __update_tracker(
            self, curr_frame: np.ndarray,
            tid: int, tracker: Any,
            x_max: int, y_max: int,
            synthetic_dets: List[DetectedObject]
    ):
        """
        Update tracker, mimic detection struct of the model.
        Returns: tid - id of tracker to delete (current)
        """
        ok, box = tracker.update(curr_frame)
        if not ok:
            # Lost tracker
            return tid

        x, y, w, h = box
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        if x1 <= self.edge_margin or y1 <= self.edge_margin or x2 >= x_max or y2 >= y_max:
            return tid

        info = self.tracked_objects.get(tid, None)
        if info is None:
            # No stored meta, skip
            return None

        # Synthetic detection labeled by lowest confidence
        synthetic_dets.append([x1, y1, x2, y2, 0.0, info["cls_idx"]])
        return None

    def _update_masks(self, curr_frame: np.ndarray) -> List[np.ndarray]:
        """
        Update all synthetic masks of tracked objects using optical flow. Returned
        order of masks is aligned with synthetic detections from _update_trackers().
        """
        synthetic_masks = []

        # Compute optical flow between prev_frame and curr_frame (grayscale)
        if self.prev_frame_gray is None:
            # No previous frame -> cannot warp masks; only attempt tracker bbox updates
            prev_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            self.prev_frame_gray = prev_gray
            flow = None
        else:
            prev_gray = self.prev_frame_gray
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            flow = self.flow.calc(prev_gray, curr_gray, None)
            # update prev_frame_gray for next call
            self.prev_frame_gray = curr_gray

        for tid, tracker in list(self.trackers.items()):
            info = self.tracked_objects.get(tid, None)

            # Warp the stored full-image mask (if available)
            mask_full = info.get("mask", None)  # mask_full is same size as image with bbox region filled
            if mask_full is not None and flow is not None:
                # Remap prev mask to current frame using flow
                # flow[...,0] = dx, flow[...,1] = dy
                H, W = mask_full.shape
                # Build meshgrid of coordinates
                grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
                map_x = (grid_x + flow[..., 0]).astype(np.float32)
                map_y = (grid_y + flow[..., 1]).astype(np.float32)
                # Remap expects maps of shape (H,W) specifying source coords for each dst pixel
                warped = cv2.remap(
                    mask_full.astype(np.float32),
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
                warped_mask_full = (warped > 0.5).astype(np.uint8)
            else:
                # No flow or no stored mask -> try to crop previous mask area and copy to new bbox
                warped_mask_full = None
            synthetic_masks.append(warped_mask_full)  # may be None

        return synthetic_masks

    def _update_trackers(self, curr_frame: np.ndarray) -> List[DetectedObject]:
        """
        Update all alive trackers, warp their stored full-image masks using optical
        flow, and return synthetic detections (format: [x1,y1,x2,y2,conf,cls]).
        Removes trackers that fail update.
        """
        synthetic_dets = []

        # Validate frame bounds
        y_max, x_max, _ = curr_frame.shape
        y_max -= self.edge_margin
        x_max -= self.edge_margin

        to_delete = []
        for tid, tracker in list(self.trackers.items()):
            to_delete_tid = self.__update_tracker(
                curr_frame, tid, tracker, x_max, y_max, synthetic_dets
            )
            if to_delete_tid is not None:
                to_delete.append(to_delete_tid)

        # Remove lost trackers
        for tid in to_delete:
            self.trackers.pop(tid, None)
            self.tracked_objects.pop(tid, None)

        return synthetic_dets

    def _rebuild_trackers_from_dets(
            self, image: np.ndarray,
            dets_nms: List[DetectedObject],
            masks_nms: Optional[List[Optional[np.ndarray]]] = None
    ) -> None:
        """
        Recreate trackers from final NMS detections.
        dets_nms: list of [x1,y1,x2,y2,conf,cls]
        masks_nms: list of mask_full (H,W) or list aligned (can be None)
        """
        # Clear existing trackers (caller handles re-identification if needed)
        self.trackers.clear()
        self.tracked_objects.clear()
        self.next_track_id = 0

        for i, det in enumerate(dets_nms):
            # Reinit trackers
            x1, y1, x2, y2, conf, cls_idx = det
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            tracker = self._create_tracker()
            tracker.init(image, (int(x1), int(y1), int(w), int(h)))

            tid = self.next_track_id
            self.next_track_id += 1

            # Build full-image mask if mask provided for this detection
            mask_full = None
            if masks_nms is not None and i < len(masks_nms):
                m = masks_nms[i]
                if m is not None:
                    # Expect m to be full-image mask already; if provided as bbox crop, paste into full size
                    if m.shape == image.shape[:2]:
                        mask_full = m.astype(np.uint8)
                    else:
                        # Assume m is bbox-local mask: paste it into full mask
                        mf = np.zeros(image.shape[:2], dtype=np.uint8)
                        bx1, by1, bx2, by2 = int(x1), int(y1), int(x2), int(y2)
                        try:
                            mf[by1:by2, bx1:bx2] = cv2.resize(m, (bx2-bx1, by2-by1), interpolation=cv2.INTER_NEAREST)
                            mask_full = mf
                        except Exception:
                            mask_full = None

            self.trackers[tid] = tracker
            self.tracked_objects[tid] = {
                "cls_idx": int(cls_idx),
                "mask": mask_full,
                "bbox": (int(x1), int(y1), int(x2), int(y2))
            }

    def predict(
            self, image: np.ndarray,
            timestamp: Optional[float] = None
    ) -> List[DetectedObject]:
        """
        Perform Yolo inference, use surrounding average
        gradients and apply CSRT trackers.
        """
        import time
        start_time = time.time()

        # Run YOLO tiled
        results = self._run_yolo_tiled(image)
        logger.debug(f"Yolo inference time: {time.time() - start_time:.3f} sec")

        # Build dets + masks lists from YOLO results
        dets, masks = self._process_results(image, results)

        # Append synthetic detections and masks to lists
        synthetic_dets = self._update_trackers(image)
        dets.extend(synthetic_dets)
        if masks is not None:
            synthetic_masks = self._update_masks(image)
            masks.extend(synthetic_masks)

        # Run NMS on combined list (nms returns kept dets and kept masks)
        dets_nms, masks_nms = self.non_max_suppression(
            dets,
            masks=masks
        )

        # Rebuild trackers from final detections (use masks_nms when available)
        self._rebuild_trackers_from_dets(image, dets_nms, masks_nms)

        # Convert final dets to DetectedObject list (masks taken from masks_nms per det)
        detected_objects = self._convert_results(image, dets_nms, masks_nms, timestamp)

        # Apply background-variance filter if desired
        return self.filter_by_background_variance(image, detected_objects)


if __name__ == "__main__":
    from src.offline_utils.frame_source import FrameSource
    from src.cv_module.visualization import plot_one_box

    im_path = "/sdb-disk/vyzai/data/image_source/sykhiv/IMG_6947.jpeg"
    image = cv2.imread(im_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    data_path = "/sdb-disk/vyzai/data/pxi_source/2024.12.02_Cityscape/2024-12-02_17-22-12/pxi/"
    data_source = FrameSource(data_path, 0, 4)
    data_tensor = next(data_source)
    image = data_tensor["view_img"]

    config_path = "configs/project_la/people_cars_detector.yaml"
    models_dir = "models/general"
    device = "cuda"
    processed_objects = ["person", "car", "truck"]

    detector = YoloDetector(config_path, models_dir, device,
                            processed_objects)

    detected_objects = detector.predict(image)

    for detected_object in detected_objects:
        plot_one_box(detected_object.bbox, image, (125, 43, 63),
                     label=detected_object.name)

    cv2.imwrite("viz_detect.png", image)
