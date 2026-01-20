import argparse
import os
import pathlib
import sys
import time
from datetime import datetime

import cv2 as cv
from loguru import logger

# Ensure project root is on sys.path when running this script directly
THIS_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.offline_utils.frame_source import FrameSource
from src.drone_rt.pipeline import DroneRTPipeline, RTPipelineConfig


def draw_tracks(frame, tracks):
    """Draw simple bbox + ID overlay for tracked drones."""
    vis = frame.copy()
    for tr in tracks:
        x1, y1, x2, y2 = tr.bbox
        cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID {tr.track_id} ({tr.score:.2f})"
        cv.putText(
            vis,
            label,
            (x1 + 5, y1 + 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return vis


def process(source_path: pathlib.Path,
            pos: int | None,
            max_frames: int | None,
            device: str = "cuda",
            detection_interval: int = 5):
    """Offline demo runner for lightweight realtime drone pipeline."""

    logger.add(
        f"logs/drone_rt_{source_path.stem}_{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}.log",
        level="INFO",
    )

    source = FrameSource(str(source_path), pos, max_frames)
    im_size = source.im_size

    # Pipeline config
    det_cfg_path = "configs/project_drones/drone_rt_detector.yaml"
    trk_cfg_path = "configs/project_drones/drone_rt_tracker.yaml"

    # Allow overriding device at runtime
    # (detector config has its own device field; we override it via env var)
    os.environ.setdefault("TORCH_DEVICE", device)

    pipeline_cfg = RTPipelineConfig(
        detector_cfg_path=det_cfg_path,
        tracker_cfg_path=trk_cfg_path,
        detection_interval=detection_interval,
    )
    pipeline = DroneRTPipeline(pipeline_cfg)

    save_path = pathlib.Path("output")
    save_path.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    save_video_path = save_path / f"drone_rt_{source_path.stem}_{ts}.mp4"

    video_out = cv.VideoWriter(
        str(save_video_path),
        cv.VideoWriter.fourcc(*"mp4v"),
        24,
        (im_size[1], im_size[0]),
        True,
    )
    assert video_out.isOpened()

    total_start = time.time()
    n_frames = 0

    for data_tensor in source:
        n_frames += 1
        logger.info("==================================================")
        logger.info(f"frame={source.i_frame} pos={source.pos} t={source.pos_sec:.3f}s")

        t0 = time.time()
        data_tensor.convert_to_numpy()
        frame = data_tensor.view_img  # RGB uint8
        ts_frame = float(getattr(data_tensor, "created_at", 0.0))

        tracks = pipeline.process_frame(frame, ts_frame)
        dt = time.time() - t0
        logger.info(f"Frame processed in {dt:.3f} s, tracks={len(tracks)}")

        frame_out = draw_tracks(frame, tracks)
        frame_out = cv.cvtColor(frame_out, cv.COLOR_RGB2BGR)
        video_out.write(frame_out)

    video_out.release()

    total_time = time.time() - total_start
    fps = n_frames / total_time if total_time > 0 else 0.0
    logger.info(f"Processed {n_frames} frames in {total_time:.3f} s -> {fps:.3f} FPS")
    logger.info(f"Saved realtime drone demo video to {save_video_path}")


def main():
    parser = argparse.ArgumentParser(description="Realtime drone RT pipeline demo (PXI or video)")
    parser.add_argument("source_path", type=pathlib.Path, help="Video file or PXI directory")
    parser.add_argument("--pos", type=int, default=None, help="Starting position")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--device", type=str, default="cuda", help="Device for models: 'cuda' or 'cpu'")
    parser.add_argument(
        "--detection-interval",
        type=int,
        default=5,
        help="Run detector every N frames",
    )
    args = parser.parse_args()

    process(
        args.source_path,
        args.pos,
        args.max_frames,
        device=args.device,
        detection_interval=args.detection_interval,
    )


if __name__ == "__main__":
    main()
