from pathlib import Path
import cv2
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from loguru import logger

from src.offline_utils.frame_source import FrameSource
from src.data_pixel_tensor import ELEMENTS_NAMES
from configs.data_pixel_tensor import DATA_PIXEL_TENSOR_BACKEND


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, default=Path("output"))
    parser.add_argument("--start-pos", default=0, type=int)
    parser.add_argument("--end-pos", default=None, type=int)
    parser.add_argument("--element", default="view_img", choices=ELEMENTS_NAMES, type=str)
    parser.add_argument("--bit-depth", default=None, type=int)
    parser.add_argument("--max-frames", default=None, type=int)
    parser.add_argument("--step", default=1, type=int)
    parser.add_argument("--downscale", default=None, type=float)
    parser.add_argument("--format", default="png", choices=["png", "jpg"], type=str)
    parser.add_argument(
        "--show-frame-idx",
        action="store_true",
        help="Draw the (0-based) frame index in the top-left corner."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of parallel workers (threads)."
    )
    parser.add_argument(
        "--index-file",
        type=Path,
        default=None,
        help="Path to a text file containing frame indices to extract, one per line. If provided, overrides start-pos, end-pos, step, and max-frames."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    source = FrameSource(args.data_path, args.start_pos, None)

    if args.end_pos is not None:
        indices = range(args.start_pos, args.end_pos, args.step)
    else:
        indices = range(args.start_pos, len(source), args.step)

    if args.max_frames is not None:
        # range supports slicing; cast to list for stable length & tqdm total
        indices = list(indices)[:args.max_frames]
    else:
        indices = list(indices)

    if args.index_file:
        with open(args.index_file) as f:
            indices = f.readlines()
            indices = [int(idx.strip()) for idx in indices]

    frames_save_path = args.save_path / f"{args.data_path.stem}"
    frames_save_path.mkdir(parents=True, exist_ok=True)

    h, w = source.im_size
    if args.downscale:
        h = int(h * args.downscale)
        w = int(w * args.downscale)

    if args.bit_depth is not None:
        max_value = 2 ** args.bit_depth
    else:
        # If start_pos > 0 this still reads absolute index 0 from source;
        # adjust if your FrameSource expects local indexing.
        max_value = source[0].max_value

    def process_and_save(img_idx: int, img_format: str = "png"):
        try:
            filename = frames_save_path / f"{args.data_path.stem}-{img_idx:03d}.{img_format}"
            if filename.exists():
                return True, img_idx, None

            data_tensor = source[img_idx]
            frame = data_tensor[args.element]

            if DATA_PIXEL_TENSOR_BACKEND == 'torch':
                frame = frame.cpu().numpy()

            # normalize 16-bit to 8-bit if needed
            if frame.dtype == np.uint16:
                frame = (frame.astype(np.float32) / max_value) * 255.0
                frame = frame.astype(np.uint8)

            # ensure 3-channel RGB then BGR for OpenCV
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if args.downscale:
                frame = cv2.resize(frame, (w, h))

            # Optional: draw frame index
            if args.show_frame_idx:
                text = f"{img_idx}"
                font_scale = max(0.5, min(2.0, h / 720.0))
                thickness = 2
                org = (10, int(30 * font_scale + 10))
                cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            cv2.imwrite(str(filename), frame)
            return True, img_idx, None
        except Exception as e:
            return False, img_idx, e

    # Parallel execution with progress bar
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_and_save, idx, args.format): idx for idx in indices}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing frames"):
            ok, idx, err = fut.result()
            if not ok:
                failures.append((idx, err))

    if failures:
        for idx, err in failures[:10]:
            logger.error(f"Frame {idx} failed: {err}")
        logger.error(f"{len(failures)} frames failed in total.")

    logger.info("Done.")

if __name__ == "__main__":
    main()
