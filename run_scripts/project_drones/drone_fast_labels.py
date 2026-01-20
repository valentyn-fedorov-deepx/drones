import argparse
import pathlib
import time
from typing import List, Optional

import cv2 as cv
import numpy as np
from loguru import logger
from omegaconf import OmegaConf

from src.offline_utils.frame_source import FrameSource
from src.drone_pipeline.detector_yolov8 import YoloV8Detector
from src.drone_pipeline.tracker_csrt_cpp import CppCSRTTracker
from src.drone_pipeline.interfaces import Detection, TrackedObjectState
from src.cv_module.basic_object import BasicObjectWithDistance
from src.cv_module.distance_measurers.width_measurer import WidthDistanceMeasurer
from src.cv_module.visualization import plot_object_with_distance
from src.utils.common import resource_path


def _load_measurer(
    det_cfg_path: str,
    manager_cfg_path: str,
    im_h: int,
    im_w: int,
) -> WidthDistanceMeasurer:
    det_cfg = OmegaConf.load(resource_path(det_cfg_path))
    manager_cfg = OmegaConf.load(resource_path(manager_cfg_path))

    focal_length_mm = float(manager_cfg.get("focal_length_mm", 28))
    sensor_ratio = float(manager_cfg.get("sensor_ratio", 0.00345))
    focal_length_px = focal_length_mm / sensor_ratio

    drone_width = float(det_cfg.get("drone_width_meters", 0.221))
    return WidthDistanceMeasurer(
        focal_length=focal_length_px,
        im_size=(im_h, im_w),
        base_width_in_meters=drone_width,
        use_tracker=True,
    )


def _rescale_bbox(
    bbox: tuple[int, int, int, int],
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Rescale bbox from src (H,W) to dst (H,W)."""
    sh, sw = src_shape
    dh, dw = dst_shape
    if sh == dh and sw == dw:
        return bbox
    sx = dw / float(sw)
    sy = dh / float(sh)
    x1, y1, x2, y2 = bbox
    return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))


def run_fast_pipeline(
    source_path: pathlib.Path,
    device: str = "cuda",
    detection_interval: int = 10,
    frame_scale: float = 0.5,
    conf: Optional[float] = None,
    fallback_conf: Optional[float] = 0.2,
    fullres_fallback: bool = True,
    motion_pred_frames: int = 30,
    motion_pred_alpha: float = 0.6,
    output_grace_frames: int = 45,
    bbox_smooth_alpha: float = 0.6,
    preview_scale: float = 1.0,
    no_video: bool = False,
):
    """High-FPS drone pipeline: YOLOv8 (TensorRT) + C++ CSRT + distance labels.

    - YOLO runs every N frames (detection_interval).
    - CSRT tracks every frame to keep ID/box stable when detection misses.
    - Distance labels (Xo/Yo/Zo/Range/V) are drawn with plot_object_with_distance().
    """

    logger.add(
        f"logs/drone_fast_labels_{source_path.stem}_{time.strftime('%Y-%m-%d__%H-%M-%S')}.log",
        level="INFO",
    )

    # Frame source
    source = FrameSource(str(source_path), None, None)
    shape = source.im_size
    if len(shape) == 3:
        im_h, im_w, _ = shape
    elif len(shape) == 2:
        im_h, im_w = shape
    else:
        raise ValueError(f"Unexpected frame shape from FrameSource.im_size: {shape}")

    # Detector config (TensorRT engine already configured here)
    det_cfg_path = "configs/project_drones/drone_detector.yaml"
    models_dir = "models/drones_hf/yolov11x/weight"
    detector = YoloV8Detector(
        config_path=det_cfg_path,
        models_dir=models_dir,
        device=device,
        processed_labels=["drone"],
    )
    base_conf = detector.get_conf()
    det_conf = base_conf if conf is None else float(conf)

    # Tracker (C++ CSRT, single drone)
    tracker = CppCSRTTracker(
        scale=1.0,
        output_grace_frames=output_grace_frames,
        motion_pred_frames=motion_pred_frames,
        motion_pred_alpha=motion_pred_alpha,
    )

    # Distance measurer (fast width-based)
    measurer = _load_measurer(
        det_cfg_path=det_cfg_path,
        manager_cfg_path="configs/project_drones/manager.yaml",
        im_h=im_h,
        im_w=im_w,
    )

    # Output video
    save_path = pathlib.Path("output")
    save_path.mkdir(exist_ok=True)
    ts = time.strftime("%Y-%m-%d__%H-%M-%S")
    out_path = save_path / f"drone_fast_labels_{source_path.stem}_{ts}.mp4"

    writer = None
    if not no_video:
        fourcc = cv.VideoWriter.fourcc(*"mp4v")
        writer = cv.VideoWriter(str(out_path), fourcc, 24, (im_w, im_h), True)
        assert writer.isOpened()

    n_frames = 0
    t_start = time.time()
    had_track = False

    smooth_boxes: dict[int, np.ndarray] = {}

    for frame_idx, dpt in enumerate(source, start=1):
        n_frames += 1
        dpt.convert_to_numpy()
        frame_full = dpt.view_img  # RGB uint8
        if not isinstance(frame_full, np.ndarray):
            try:
                frame_full = frame_full.detach().cpu().numpy()
            except AttributeError:
                frame_full = frame_full.cpu().numpy()
        ts_frame = float(getattr(dpt, "created_at", time.time()))

        # Downscale for detector + tracker (speed)
        if frame_scale != 1.0:
            fh, fw = frame_full.shape[:2]
            small_w = max(1, int(fw * frame_scale))
            small_h = max(1, int(fh * frame_scale))
            frame = cv.resize(frame_full, (small_w, small_h), interpolation=cv.INTER_AREA)
        else:
            frame = frame_full

        # Detection (every N frames or if track lost)
        detections: List[Detection] = []
        force_detect = not had_track
        if frame_idx % detection_interval == 1 or force_detect:
            dets = detector.detect_with_overrides(frame, ts_frame, conf=det_conf)
            if not dets and fallback_conf is not None and fallback_conf < det_conf:
                dets = detector.detect_with_overrides(frame, ts_frame, conf=fallback_conf)
                if dets:
                    logger.info(
                        f"Fallback detect used (conf={fallback_conf:.2f}), dets={len(dets)}"
                    )
            # Optional full-res fallback for small drones on complex background
            if not dets and fullres_fallback and frame_scale != 1.0:
                dets_full = detector.detect_with_overrides(frame_full, ts_frame, conf=fallback_conf or det_conf)
                if dets_full:
                    # Map detections to scaled frame for tracker
                    mapped: List[Detection] = []
                    for d in dets_full:
                        sx_bbox = _rescale_bbox(d.bbox, frame_full.shape[:2], frame.shape[:2])
                        mapped.append(
                            Detection(
                                bbox=sx_bbox,
                                score=d.score,
                                label=d.label,
                                timestamp=d.timestamp,
                                frame_index=d.frame_index,
                            )
                        )
                    dets = mapped
                    logger.info("Full-res fallback detect used")
            if dets:
                detections = [max(dets, key=lambda d: d.score)]

        # Tracking
        states: List[TrackedObjectState] = tracker.update(frame, ts_frame, detections)

        # Visualization
        vis = frame_full.copy()
        if states:
            objects: List[BasicObjectWithDistance] = []
            for s in states:
                # Rescale bbox to full-res if needed
                x1, y1, x2, y2 = _rescale_bbox(
                    s.bbox,
                    src_shape=frame.shape[:2],
                    dst_shape=frame_full.shape[:2],
                )
                bbox_arr = np.array([x1, y1, x2, y2], dtype=float)
                if bbox_smooth_alpha > 0:
                    prev = smooth_boxes.get(s.track_id)
                    if prev is None:
                        smooth = bbox_arr
                    else:
                        smooth = bbox_smooth_alpha * bbox_arr + (1.0 - bbox_smooth_alpha) * prev
                    smooth_boxes[s.track_id] = smooth
                    x1, y1, x2, y2 = smooth.astype(int).tolist()

                obj = BasicObjectWithDistance(
                    id=s.track_id,
                    bbox=(x1, y1, x2, y2),
                    conf=float(s.score),
                    name=str(s.label),
                    timestamp=float(s.timestamp),
                )
                objects.append(obj)

            # Compute measurements (X/Y/Z/Range/Velocity)
            measurements = measurer.process(objects)
            for obj in objects:
                obj.meas = measurements.get(obj.id)

            vis = plot_object_with_distance(vis, objects)

        if writer is not None:
            out_frame = vis
            if preview_scale != 1.0:
                oh, ow = out_frame.shape[:2]
                out_frame = cv.resize(
                    out_frame,
                    (max(1, int(ow * preview_scale)), max(1, int(oh * preview_scale))),
                    interpolation=cv.INTER_AREA,
                )
            writer.write(cv.cvtColor(out_frame, cv.COLOR_RGB2BGR))

        had_track = bool(states)

    if writer is not None:
        writer.release()

    t_total = time.time() - t_start
    fps = n_frames / t_total if t_total > 0 else 0.0
    logger.info(f"FAST labels processed {n_frames} frames in {t_total:.3f}s -> {fps:.3f} FPS")
    if writer is not None:
        logger.info(f"Saved fast labels video to {out_path}")



def main():
    parser = argparse.ArgumentParser(
        description="High-FPS drone demo with TensorRT + CSRT + distance labels"
    )
    parser.add_argument("source_path", type=pathlib.Path, help="Video file or PXI directory")
    parser.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument(
        "--detection-interval",
        type=int,
        default=10,
        help="Run YOLO every N frames (when track is stable)",
    )
    parser.add_argument(
        "--frame-scale",
        type=float,
        default=0.5,
        help="Global downscale factor for full frame (YOLO + CSRT)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detector confidence (overrides config)",
    )
    parser.add_argument(
        "--fallback-conf",
        type=float,
        default=0.2,
        help="Lower confidence for fallback detection when no boxes are found",
    )
    parser.add_argument(
        "--fullres-fallback",
        action="store_true",
        help="Run a one-off full-res detection when scaled detect fails",
    )
    parser.add_argument(
        "--motion-pred-frames",
        type=int,
        default=30,
        help="How many frames to keep motion prediction when tracking is lost",
    )
    parser.add_argument(
        "--motion-pred-alpha",
        type=float,
        default=0.6,
        help="EMA smoothing for motion prediction velocity",
    )
    parser.add_argument(
        "--output-grace-frames",
        type=int,
        default=45,
        help="How long to keep output when tracker is lost",
    )
    parser.add_argument(
        "--bbox-smooth-alpha",
        type=float,
        default=0.6,
        help="EMA smoothing for bbox (0 disables)",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=1.0,
        help="Downscale output video for faster writing",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Do not save output video, only measure FPS",
    )
    args = parser.parse_args()

    run_fast_pipeline(
        args.source_path,
        device=args.device,
        detection_interval=args.detection_interval,
        frame_scale=args.frame_scale,
        conf=args.conf,
        fallback_conf=args.fallback_conf,
        fullres_fallback=args.fullres_fallback,
        motion_pred_frames=args.motion_pred_frames,
        motion_pred_alpha=args.motion_pred_alpha,
        output_grace_frames=args.output_grace_frames,
        bbox_smooth_alpha=args.bbox_smooth_alpha,
        preview_scale=args.preview_scale,
        no_video=args.no_video,
    )


if __name__ == "__main__":
    main()
