import argparse
import pathlib

import cv2 as cv
from loguru import logger

from src.offline_utils.frame_source import FrameSource


def pxi_to_video(source_path: pathlib.Path, out_path: pathlib.Path, fps: int | None = None):
    """Convert PXI/directory source to MP4 using FrameSource.view_img.

    This is an offline conversion helper so we can later run the realtime
    pipeline on plain video instead of heavy PXI preprocessing.
    """

    logger.info(f"PXI->Video: source={source_path}")

    # For PXI directories we use FrameSource in files_mode
    # For video files it will use OpenCV's VideoCapture.
    fs = FrameSource(str(source_path), None, None)

    im_size = fs.im_size  # (H,W,...) from DataPixelTensor.raw
    if len(im_size) == 3:
        h, w, _ = im_size
    elif len(im_size) == 2:
        h, w = im_size
    else:
        raise ValueError(f"Unexpected im_size from FrameSource: {im_size}")

    if fps is None:
        # Try to estimate FPS for PXI directories; fall back to 25 otherwise.
        try:
            fps_val = fs.estimate_fps()
            logger.info(f"Estimated FPS from PXI header: {fps_val}")
            if fps_val is None or fps_val <= 0:
                raise ValueError(f"Invalid estimated FPS: {fps_val}")
        except Exception as e:
            logger.warning(f"Failed to estimate valid FPS, fallback to 25: {e}")
            fps_val = 25
    else:
        fps_val = fps

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv.VideoWriter.fourcc(*"mp4v")
    writer = cv.VideoWriter(str(out_path), fourcc, fps_val, (w, h), True)
    assert writer.isOpened(), f"Failed to open VideoWriter for {out_path}"

    n_frames = 0
    for dpt in fs:
        dpt.convert_to_numpy()
        frame = dpt.view_img  # RGB uint8
        bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        writer.write(bgr)
        n_frames += 1

    writer.release()
    logger.info(f"Saved {n_frames} frames to {out_path} @ {fps_val} FPS")


def main():
    parser = argparse.ArgumentParser(description="Convert PXI/directory to MP4 using FrameSource")
    parser.add_argument("source_path", type=pathlib.Path, help="PXI file or directory")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=None, help="Override FPS (otherwise estimated or 25)")
    args = parser.parse_args()

    src = args.source_path
    if args.out is None:
        out_dir = pathlib.Path("output")
        out_dir.mkdir(exist_ok=True)
        out_name = src.name.rstrip("/\\") + "_pxi.mp4"
        out_path = out_dir / out_name
    else:
        out_path = args.out

    pxi_to_video(src, out_path, fps=args.fps)


if __name__ == "__main__":
    main()
