import argparse
import pathlib
import time

import cv2 as cv
from loguru import logger

from src.offline_utils.frame_source import FrameSource
from src.drone_pipeline.detector_yolov8 import YoloV8Detector
from src.drone_pipeline.tracker_csrt_cpp import CppCSRTTracker
from src.drone_pipeline.interfaces import Detection
from src.utils.common import resource_path


def run_fast_pipeline(
    source_path: pathlib.Path,
    device: str = "cuda",
    detection_interval: int = 60,
    scale: float = 0.4,
    frame_scale: float = 0.5,
):
    """Minimal high-FPS drone pipeline: YOLOv8 + C++ CSRT (single drone).

    - YOLO працює рідко (кожні `detection_interval` кадрів) на GPU.
    - CppCSRTTracker(scale=...) трекає кожен кадр по даунсемпленому зображенню.
    - Без сегментації та класифікації, тільки bbox + ID.
    """

    logger.add(
        f"logs/drone_fast_{source_path.stem}_{time.strftime('%Y-%m-%d__%H-%M-%S')}.log",
        level="INFO",
    )

    # Ініціалізуємо джерело кадрів
    source = FrameSource(str(source_path), None, None)
    shape = source.im_size
    if len(shape) == 3:
        im_h, im_w, _ = shape
    elif len(shape) == 2:
        im_h, im_w = shape
    else:
        raise ValueError(f"Unexpected frame shape from FrameSource.im_size: {shape}")

    # Завантажуємо YOLOv8 детектор дронів
    det_cfg_path = "configs/project_drones/drone_detector.yaml"
    models_dir = "models/drones_hf/yolov11x/weight"
    detector = YoloV8Detector(
        config_path=det_cfg_path,
        models_dir=models_dir,
        device=device,
        processed_labels=["drone"],
    )

    # C++ CSRT трекер (single-drone)
    tracker = CppCSRTTracker(scale=1.0)

    save_path = pathlib.Path("output")
    save_path.mkdir(exist_ok=True)
    ts = time.strftime("%Y-%m-%d__%H-%M-%S")
    out_path = save_path / f"drone_fast_{source_path.stem}_cpp_{ts}.mp4"

    fourcc = cv.VideoWriter.fourcc(*"mp4v")
    writer = cv.VideoWriter(str(out_path), fourcc, 24, (im_w, im_h), True)
    assert writer.isOpened()

    n_frames = 0
    t_start = time.time()
    had_track = False

    for frame_idx, dpt in enumerate(source, start=1):
        n_frames += 1
        dpt.convert_to_numpy()
        frame_full = dpt.view_img  # RGB uint8 (full-res)
        ts_frame = float(getattr(dpt, "created_at", 0.0))

        # Глобальний даунскейл кадру для всього пайплайна (YOLO + CSRT)
        if frame_scale != 1.0:
            fh, fw = frame_full.shape[:2]
            small_w = max(1, int(fw * frame_scale))
            small_h = max(1, int(fh * frame_scale))
            frame = cv.resize(frame_full, (small_w, small_h), interpolation=cv.INTER_AREA)
        else:
            frame = frame_full

        # Раз на detection_interval кадрів або при втраті треку запускаємо YOLO
        detections: list[Detection] = []
        force_detect = not had_track  # якщо ще не було стабільного треку
        if frame_idx % detection_interval == 1 or force_detect:
            t_det0 = time.time()
            dets = detector.detect(frame, ts_frame)
            t_det1 = time.time()
            logger.info(
                f"YOLO detect() took {(t_det1 - t_det0)*1000:.2f} ms, dets={len(dets)}"
            )

            if dets:
                # Для одного дрона беремо детекцію з найбільшим score
                best = max(dets, key=lambda d: d.score)
                detections = [best]

        t_tr0 = time.time()
        states = tracker.update(frame, ts_frame, detections)
        t_tr1 = time.time()
        logger.info(
            f"frame={frame_idx} tracking took {(t_tr1 - t_tr0)*1000:.2f} ms, "
            f"states={states}"
        )

        # Візуалізація: рескейл bbox назад у full-res, якщо кадр був зменшений
        vis = frame_full.copy()
        if states:
            s = states[0]
            x1, y1, x2, y2 = s.bbox
            if frame_scale != 1.0:
                fh, fw = frame_full.shape[:2]
                sh, sw = frame.shape[:2]
                sx = fw / float(sw)
                sy = fh / float(sh)
                x1 = int(x1 * sx)
                y1 = int(y1 * sy)
                x2 = int(x2 * sx)
                y2 = int(y2 * sy)
            cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(
                vis,
                f"ID {s.track_id}",
                (x1 + 5, y1 + 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        writer.write(cv.cvtColor(vis, cv.COLOR_RGB2BGR))
        had_track = bool(states)

    writer.release()
    t_total = time.time() - t_start
    fps = n_frames / t_total if t_total > 0 else 0.0
    logger.info(f"FAST pipeline processed {n_frames} frames in {t_total:.3f}s -> {fps:.3f} FPS")
    logger.info(f"Saved fast demo video to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="High-FPS drone tracker demo (YOLO + C++ CSRT)",
    )
    parser.add_argument("source_path", type=pathlib.Path, help="Video file or PXI directory")
    parser.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument(
        "--detection-interval",
        type=int,
        default=60,
        help="Run YOLO every N frames (when track is stable)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.0,
        help="(DEPRECATED) CSRT internal scale, keep 0.0 to use frame_scale only",
    )
    parser.add_argument(
        "--frame-scale",
        type=float,
        default=0.5,
        help="Global downscale factor for full frame (YOLO + CSRT)",
    )
    args = parser.parse_args()

    run_fast_pipeline(
        args.source_path,
        device=args.device,
        detection_interval=args.detection_interval,
        scale=args.scale,
        frame_scale=args.frame_scale,
    )


if __name__ == "__main__":
    main()
import pathlib
import time

import cv2 as cv
from loguru import logger

from src.offline_utils.frame_source import FrameSource
from src.drone_pipeline.detector_yolov8 import YoloV8Detector
from src.drone_pipeline.tracker_csrt_cpp import CppCSRTTracker
from src.drone_pipeline.interfaces import Detection
from src.utils.common import resource_path


def run_fast_pipeline(
    source_path: pathlib.Path,
    device: str = "cuda",
    detection_interval: int = 60,
    scale: float = 0.4,
    frame_scale: float = 0.5,
):
    """Minimal high-FPS drone pipeline: YOLOv8 + C++ CSRT (single drone).

    - YOLO працює рідко (кожні `detection_interval` кадрів) на GPU.
    - CppCSRTTracker(scale=...) трекає кожен кадр по даунсемпленому зображенню.
    - Без сегментації та класифікації, тільки bbox + ID.
    """

    logger.add(
        f"logs/drone_fast_{source_path.stem}_{time.strftime('%Y-%m-%d__%H-%M-%S')}.log",
        level="INFO",
    )

    # Ініціалізуємо джерело кадрів
    source = FrameSource(str(source_path), None, None)
    # Підтримуємо як 2D (H, W), так і 3D (H, W, C) форми кадру.
    _shape = source.im_size
    if len(_shape) == 3:
        im_h, im_w, _ = _shape
    elif len(_shape) == 2:
        im_h, im_w = _shape
    else:
        raise ValueError(f"Unexpected frame shape from FrameSource.im_size: {_shape}")

    # Завантажуємо YOLOv8 детектор дронів
    det_cfg_path = "configs/project_drones/drone_detector.yaml"
    models_dir = "models/drones_hf/yolov11x/weight"
    detector = YoloV8Detector(
        config_path=det_cfg_path,
        models_dir=models_dir,
        device=device,
        processed_labels=["drone"],
    )

    # C++ CSRT трекер.
    # Для чистоти експерименту з глобальним даунскейлом кадру тримаємо scale=1.0,
    # щоб не дублювати масштабування (frame_scale вже зменшує розмір кадру).
    tracker = CppCSRTTracker(scale=1.0)

    save_path = pathlib.Path("output")
    save_path.mkdir(exist_ok=True)
    ts = time.strftime("%Y-%m-%d__%H-%M-%S")
    out_path = save_path / f"drone_fast_{source_path.stem}_cpp_{ts}.mp4"

    fourcc = cv.VideoWriter.fourcc(*"mp4v")
    writer = cv.VideoWriter(str(out_path), fourcc, 24, (im_w, im_h), True)
    assert writer.isOpened()

    n_frames = 0
    t_start = time.time()
    had_track = False

    for frame_idx, dpt in enumerate(source, start=1):
        n_frames += 1
        dpt.convert_to_numpy()
        frame_full = dpt.view_img  # RGB uint8 (full-res)
        ts_frame = float(getattr(dpt, "created_at", 0.0))

        # Глобальний даунскейл кадру для всього пайплайна (YOLO + CSRT)
        if frame_scale != 1.0:
            fh, fw = frame_full.shape[:2]
            small_w = max(1, int(fw * frame_scale))
            small_h = max(1, int(fh * frame_scale))
            frame = cv.resize(frame_full, (small_w, small_h), interpolation=cv.INTER_AREA)
        else:
            frame = frame_full

        # Раз на detection_interval кадрів або при втраті треку запускаємо YOLO
        detections: list[Detection] = []
        force_detect = not had_track  # якщо ще не було стабільного треку
        if frame_idx % detection_interval == 1 or force_detect:
            t_det0 = time.time()
            dets = detector.detect(frame, ts_frame)
            t_det1 = time.time()
            logger.info(f"YOLO detect() took {(t_det1 - t_det0)*1000:.2f} ms, dets={len(dets)}")

            if dets:
                # Для одного дрона беремо детекцію з найбільшим score
                best = max(dets, key=lambda d: d.score)
                detections = [best]

        t_tr0 = time.time()
        states = tracker.update(frame, ts_frame, detections)
        t_tr1 = time.time()
        logger.info(
            f"frame={frame_idx} tracking took {(t_tr1 - t_rc0)*1000:.2f} ms, "
            f"states={states}"
        )

        # Візуалізація: рескейл bbox назад у full-res, якщо кадр був зменшений
        vis = frame_full.copy()
        if states:
            s = states[0]
            x1, y1, x2, y2 = s.bbox
            if frame_scale != 1.0:
                fh, fw = frame_full.shape[:2]
                sh, sw = frame.shape[:2]
                sx = fw / float(sw)
                sy = fh / float(sh)
                x1 = int(x1 * sx)
                y1 = int(y1 * sy)
                x2 = int(x2 * sx)
                y2 = int(y2 * sy)
            cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(
                vis,
                f"ID {s.track_id}",
                (x1 + 5, y1 + 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        writer.write(cv.cvtColor(vis, cv.COLOR_RGB2BGR))

        # Оновлюємо прапорець, чи є трек
        had_track = bool(states)

    writer.release()
    t_total = time.time() - t_start
    fps = n_frames / t_total if t_total > 0 else 0.0
    logger.info(f"FAST pipeline processed {n_frames} frames in {t_total:.3f}s -> {fps:.3f} FPS")
    logger.info(f"Saved fast demo video to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="High-FPS drone tracker demo (YOLO + C++ CSRT)"
    )
    parser.add_argument("source_path", type=pathlib.Path, help="Video file or PXI directory")
    parser.add_argument("--hp", type=str, default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument(
        "--detection-interval",
        type=int,
        default=60,
        help="Run YOLO every N frames (when track is stable)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.0,
        help="(DEPRECATED) CSRT internal scale, keep 0.0 to use frame_scale only",
    )
    parser.add_argument(
        "--frame-scale",
        type=float,
        default=0.5,
        help="Global downscale factor for full frame (YOLO + CSRT)",
    )
    args = parser.parse_args()

    run_fast_pipeline(
        args.source_path,
        device=args.device,
        detection_interval=args.detection_interval,
        scale=args.scale,
        frame_scale=args.frame_scale,
    )


if __name__ == "__main__":
    main()
    source_path: pathlib.Path,
    device: str = "cuda",
    detection_interval: int = 60,
    scale: float = 0.4,
    frame_scale: float = 0.5,
    no_video: bool = False,
    tracker_backend: str = "cpp",
):
    """Minimal high-FPS drone pipeline: YOLOv8 + C++ CSRT (single drone).

    - YOLO працює рідко (кожні `detection_interval` кадрів) на GPU.
    - CppCSRTTracker(scale=...) трекає кожен кадр по даунсемпленому зображенню.
    - Без сегментації та класифікації, тільки bbox + ID.
    """

    logger.add(
        f"logs/drone_fast_{source_path.stem}_{time.strftime('%Y-%m-%d__%H-%M-%S')}.log",
        level="INFO",
    )

    # Використовуємо окремий потік для YOLO, щоб детектор не блокував головний цикл.
    executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
    pending_future: Optional[Future[Optional[Detection]]] = None
    latest_det: Optional[Detection] = None

    # Ініціалізуємо джерело кадрів
    source = FrameSource(str(source_path), None, None)
    # Підтримуємо як 2D (H, W), так і 3D (H, W, C) форми кадру.
    _shape = source.im_size
    if len(_shape) == 3:
        im_h, im_w, _ = _shape
    elif len(_shape) == 2:
        im_h, im_w = _shape
    else:
        raise ValueError(f"Unexpected frame shape from FrameSource.im_size: {_shape}")

    # Завантажуємо YOLOv8 детектор дронів
    det_cfg_path = "configs/project_drones/drone_detector.yaml"
    models_dir = "models/drones_hf/yolov11x/weight"
    detector = YoloV8Detector(
        config_path=det_cfg_path,
        models_dir=models_dir,
        device=device,
        processed_labels=["drone"],
    )

    # Вибір трекера: C++ CSRT або простий GPU template-matching трекер на Torch.
    if tracker_backend == "cpp":
        tracker = CppCSRTTracker(scale=1.0)
    elif tracker_backend == "gpu":
        tracker = GpuTemplateTracker(device=device)
    else:
        raise ValueError(f"Unknown tracker_backend: {tracker_backend}")

    save_path = pathlib.Path("output")
    save_path.mkdir(exist_ok=True)
    ts = time.strftime("%Y-%m-%d__%H-%M-%S")
    out_path = save_path / f"drone_fast_{source_path.stem}_cpp_{ts}.mp4"

    writer = None
    if not no_video:
        fourcc = cv.VideoWriter.fourcc(*"mp4v")
        writer = cv.VideoWriter(str(out_path), fourcc, 24, (im_w, im_h), True)
        assert writer.isOpened()

    n_frames = 0
    t_start = time.time()
    had_track = False

    # Для оцінки напрямку руху запам'ятовуємо центр bbox з попереднього кадру.
    last_centers: dict[int, tuple[float, float]] = {}

    for frame_idx, dpt in enumerate(source, start=1):
        n_frames += 1
        # Convert any cached tensors in DataPixelTensor to NumPy (CPU).
        dpt.convert_to_numpy()
        frame_full = dpt.view_img  # RGB uint8 (full-res) or torch.Tensor if just computed
        # If Torch backend produced a torch.Tensor that is not yet in NumPy, convert here.
        if not isinstance(frame_full, np.ndarray):
            try:
                frame_full = frame_full.detach().cpu().numpy()
            except AttributeError:
                frame_full = frame_full.cpu().numpy()
        ts_frame = float(getattr(dpt, "created_at", 0.0))

        # Глобальний даунскейл кадру для всього пайплайна (YOLO + CSRT)
        if frame_scale != 1.0:
            fh, fw = frame_full.shape[:2]
            small_w = max(1, int(fw * frame_scale))
            small_h = max(1, int(fh * frame_scale))
            frame = cv.resize(frame_full, (small_w, small_h), interpolation=cv.INTER_AREA)
        else:
            frame = frame_full

        # Раз на detection_interval кадрів або при втраті треку ставимо задачу YOLO
        # в окремий потік, щоб не блокувати головний цикл.
        # Спочатку забираємо результат попередньої задачі, якщо вона завершилась.
        if pending_future is not None and pending_future.done():
            try:
                det_res = pending_future.result()
            except Exception as e:
                logger.warning(f"Async YOLO detect() failed: {e}")
                det_res = None
            latest_det = det_res
            pending_future = None

        detections: list[Detection] = []
        if latest_det is not None:
            detections = [latest_det]

        force_detect = not had_track  # якщо ще не було стабільного треку
        should_request = (frame_idx % detection_interval == 1) or force_detect

        if should_request and pending_future is None:
            frame_copy = frame.copy()
            ts_copy = ts_frame

            def _detect_job() -> Optional[Detection]:
                t_det0 = time.time()
                dets = detector.detect(frame_copy, ts_copy)
                t_det1 = time.time()
                logger.info(
                    f"[ASYNC] YOLO detect() took {(t_det1 - t_det0)*1000:.2f} ms, dets={len(dets)}"
                )
                if not dets:
                    return None
                # Для одного дрона беремо детекцію з найбільшим score
                return max(dets, key=lambda d: d.score)

            pending_future = executor.submit(_detect_job)

        t_tr0 = time.time()
        states = tracker.update(frame, ts_frame, detections)
        t_tr1 = time.time()
        logger.info(
            f"frame={frame_idx} tracking took {(t_tr1 - t_tr0)*1000:.2f} ms, "
            f"states={states}"
        )

        # Візуалізація: рескейл bbox назад у full-res, якщо кадр був зменшений
        vis = frame_full.copy()
        if states:
            # Поки у нас один дрон, працюємо з першим треком.
            s = states[0]
            x1, y1, x2, y2 = s.bbox
            # Масштабуємо bbox назад у full-res, якщо кадр був зменшений.
            if frame_scale != 1.0:
                fh, fw = frame_full.shape[:2]
                sh, sw = frame.shape[:2]
                sx = fw / float(sw)
                sy = fh / float(sh)
                x1 = int(x1 * sx)
                y1 = int(y1 * sy)
                x2 = int(x2 * sx)
                y2 = int(y2 * sy)

            # 1) Малюємо bbox, як раніше.
            cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 2) Рахуємо центр і приблизну 2D-швидкість у пікселях за кадр.
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            vx, vy = 0.0, 0.0
            if s.track_id in last_centers:
                prev_cx, prev_cy = last_centers[s.track_id]
                vx = cx - prev_cx
                vy = cy - prev_cy
            last_centers[s.track_id] = (cx, cy)

            speed_pix = float(np.sqrt(vx ** 2 + vy ** 2))

            # 3) Формуємо лейбл у стилі старого пайплайна: ID, координати, швидкість.
            labels = [
                f"DRONE #{s.track_id}",
                f"cx: {cx:.1f} px",
                f"cy: {cy:.1f} px",
                f"V: {speed_pix:.2f} px/frame",
                "",  # порожній рядок, щоб зарезервувати місце під стрілку напряму
            ]

            ann_settings = ANNOTATION_SETTINGS.get("drone", DEFAULT_ANNOTATION_SETTING)
            vis = annotate_with_arrow(vis, (x1, y1, x2, y2), labels, ann_settings, velocity=(vx, vy))

        if writer is not None:
            writer.write(cv.cvtColor(vis, cv.COLOR_RGB2BGR))

        # Оновлюємо прапорець, чи є трек
        had_track = bool(states)

    # Коректно завершуємо executor, щоб дочекатись останніх задач (якщо є).
    executor.shutdown(wait=True)

    t_total = time.time() - t_start
    fps = n_frames / t_total if t_total > 0 else 0.0
    logger.info(f"FAST pipeline processed {n_frames} frames in {t_total:.3f}s -> {fps:.3f} FPS")
    if writer is not None:
        logger.info(f"Saved fast demo video to {out_path}")



def main():
    parser = argparse.ArgumentParser(description="High-FPS drone tracker demo (YOLO + C++ CSRT)")
    parser.add_argument("source_path", type=pathlib.Path, help="Video file or PXI directory")
    parser.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument("--detection-interval", type=int, default=60, help="Run YOLO every N frames (when track is stable)")
    parser.add_argument("--scale", type=float, default=0.0, help="(DEPRECATED) CSRT internal scale, keep 0.0 to use frame_scale only")
    parser.add_argument("--frame-scale", type=float, default=0.5, help="Global downscale factor for full frame (YOLO + CSRT)")
    parser.add_argument("--no-video", action="store_true", help="Do not save output video, only measure FPS")
    parser.add_argument(
        "--tracker-backend",
        type=str,
        default="cpp",
        choices=["cpp", "gpu"],
        help="Tracker backend: 'cpp' for C++ CSRT, 'gpu' for Torch-based template tracker",
    )
    args = parser.parse_args()

    run_fast_pipeline(
        args.source_path,
        device=args.device,
        detection_interval=args.detection_interval,
        scale=args.scale,
        frame_scale=args.frame_scale,
        no_video=args.no_video,
        tracker_backend=args.tracker_backend,
    )


if __name__ == "__main__":
    main()
