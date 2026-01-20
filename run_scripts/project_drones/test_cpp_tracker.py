import argparse
import pathlib
import time

import cv2 as cv
from loguru import logger

from src.offline_utils.frame_source import FrameSource
from src.drone_pipeline.tracker_csrt_cpp import CppCSRTTracker
from src.drone_pipeline.interfaces import Detection


def main():
    parser = argparse.ArgumentParser(description="Minimal C++ CSRT tracker test on a single sequence")
    parser.add_argument("source_path", type=pathlib.Path, help="Video file or PXI directory")
    parser.add_argument("--pos", type=int, default=300, help="Starting frame index")
    parser.add_argument("--max-frames", type=int, default=50, help="Number of frames to test")
    args = parser.parse_args()

    logger.add(
        f"logs/test_cpp_tracker_{args.source_path.stem}_{time.strftime('%Y-%m-%d__%H-%M-%S')}.log",
        level="DEBUG",
    )

    logger.info(f"sys.path for debug:")
    import sys
    for p in sys.path:
        logger.info(f"  {p}")

    logger.info("Trying direct import of csrt_tracker_ext...")
    try:
        import csrt_tracker_ext  # type: ignore
        logger.info(f"csrt_tracker_ext imported OK from {csrt_tracker_ext.__file__}")
    except Exception as e:  # pragma: no cover
        logger.exception(f"FAILED to import csrt_tracker_ext: {e}")

    logger.info("Constructing CppCSRTTracker() ...")
    try:
        tracker = CppCSRTTracker()
    except Exception as e:
        logger.exception(f"FAILED to construct CppCSRTTracker: {e}")
        return

    source = FrameSource(str(args.source_path), args.pos, args.max_frames)
    im_h, im_w, _ = source.im_size

    first_det = None

    for frame_idx, dpt in enumerate(source, start=1):
        logger.info("==================================================")
        logger.info(f"frame={source.i_frame} pos={source.pos} t={source.pos_sec:.3f}s")

        dpt.calculate_all()
        dpt.convert_to_numpy()
        frame = dpt.view_img  # RGB uint8

        # На першому кадрі беремо bbox по всьому зображенню як фейкову детекцію
        if first_det is None:
            bbox = (int(0.25 * im_w), int(0.25 * im_h), int(0.75 * im_w), int(0.75 * im_h))
            first_det = Detection(bbox=bbox, score=1.0, label="drone", timestamp=float(dpt.created_at))
            dets = [first_det]
        else:
            dets = []

        t0 = time.time()
        states = tracker.update(frame, float(dpt.created_at), dets)
        dt = time.time() - t0

        logger.info(f"CppCSRTTracker.update() took {dt*1000:.2f} ms, states={states}")

        # Візуалізуємо трек, якщо є
        vis = frame.copy()
        if states:
            s = states[0]
            x1, y1, x2, y2 = s.bbox
            cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(vis, f"ID {s.track_id}", (x1 + 5, y1 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if frame_idx == 1:
            cv.imwrite(str(pathlib.Path("output") / f"test_cpp_tracker_first_frame.png"), cv.cvtColor(vis, cv.COLOR_RGB2BGR))

    logger.info("Test finished.")


if __name__ == "__main__":
    main()
