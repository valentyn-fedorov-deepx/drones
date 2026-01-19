import argparse
import pathlib
import time
from datetime import datetime

import cv2 as cv
from loguru import logger

from src.project_managers.project_drones_manager import ProjectDronesManager
from src.offline_utils.frame_source import FrameSource


def process(source_path: pathlib.Path, pos: int | None, max_frames: int | None,
            device: str = "cuda", tracker_backend: str = "python"):
    """Offline demo runner for the project_drones pipeline."""
    logger.add(
        f"logs/drone_drones_demo_{source_path.stem}_{time.strftime('%Y-%m-%d__%H-%M-%S')}.log",
        level="INFO",
    )

    source = FrameSource(str(source_path), pos, max_frames)
    im_size = source.im_size

    logger.info(f"Using device='{device}' for ProjectDronesManager")
    logger.info(f"Using tracker_backend='{tracker_backend}' for ProjectDronesManager")
    manager = ProjectDronesManager(
        config_path="configs/project_drones/manager.yaml",
        device=device,
        tracker_backend=tracker_backend,
    )

    save_path = pathlib.Path("output")
    save_path.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    save_video_path = save_path / f"drone_drones_demo_{source_path.stem}_{ts}.mp4"

    # Video writer
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
        logger.info("==================================================")
        logger.info(f"frame={source.i_frame} pos={source.pos} t={source.pos_sec:.3f}s")
        n_frames += 1

        t0 = time.time()
        data_tensor.calculate_all()
        data_tensor.convert_to_numpy()
        manager.process(data_tensor)
        dt = time.time() - t0
        logger.info(f"Frame processed in {dt:.3f} s (fast path incl. tracking)")

        frame_out = manager.generate_vizualization_for_latest_data()
        if frame_out is None or getattr(frame_out, "size", 0) == 0:
            frame_out = data_tensor.view_img
        frame_out = cv.cvtColor(frame_out, cv.COLOR_RGB2BGR)
        video_out.write(frame_out)

    video_out.release()

    total_time = time.time() - total_start
    fps = n_frames / total_time if total_time > 0 else 0.0
    logger.info(f"Processed {n_frames} frames in {total_time:.3f} s -> {fps:.3f} FPS")
    logger.info(f"Saved drone_drones demo video to {save_video_path}")


def main():
    parser = argparse.ArgumentParser(description="Drone pipeline demo for project_drones (PXI or video)")
    parser.add_argument("source_path", type=pathlib.Path, help="Video file or PXI directory")
    parser.add_argument("--pos", type=int, default=None, help="Starting position")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--device", type=str, default="cuda", help="Device for models: 'cuda' or 'cpu'")
    parser.add_argument("--tracker-backend", type=str, default="python",
                        choices=["python", "cpp"],
                        help="Tracker backend to use: 'python' (default) or 'cpp' (C++ CSRT)")
    args = parser.parse_args()

    process(
        args.source_path,
        args.pos,
        args.max_frames,
        device=args.device,
        tracker_backend=args.tracker_backend,
    )


if __name__ == "__main__":
    main()
