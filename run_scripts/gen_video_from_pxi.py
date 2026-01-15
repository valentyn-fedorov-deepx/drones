from pathlib import Path
import cv2
from argparse import ArgumentParser
from tqdm import tqdm

from loguru import logger

from src.offline_utils.frame_source import FrameSource
from src.data_pixel_tensor import ELEMENTS_NAMES
from configs.data_pixel_tensor import DATA_PIXEL_TENSOR_BACKEND


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, default="output")
    parser.add_argument("--fps", default=None, type=int)
    parser.add_argument("--start-pos", default=0, type=int)
    parser.add_argument("--end-pos", default=None, type=int)
    parser.add_argument("--element", default="view_img", choices=ELEMENTS_NAMES, type=str)
    parser.add_argument("--bit-depth", default=None, type=int)
    parser.add_argument("--max-frames", default=None, type=int)
    parser.add_argument("--step", default=1, type=int)
    parser.add_argument("--downscale", default=None, type=float)
    parser.add_argument(
        "--show-frame-idx",
        action="store_true",
        help="Draw the (0-based) frame index in the top-left corner."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    source = FrameSource(args.data_path, args.start_pos, None)
    if args.end_pos is not None:
        indices = range(args.start_pos, args.end_pos, args.step)
    else:
        indices = range(args.start_pos, len(source), args.step)

    indices = indices[:args.max_frames]


    args.save_path.mkdir(parents=True, exist_ok=True)
    video_save_path = str(args.save_path / f"{args.data_path.stem}.mp4")

    h, w = source.im_size
    if args.downscale:
        h = int(h * args.downscale)
        w = int(w * args.downscale)

    if args.fps is None:
        fps = source.estimate_fps()
        logger.info(f"Estimated fps is {fps}")
    else:
        fps = args.fps

    new_video = cv2.VideoWriter(
        video_save_path,
        cv2.VideoWriter.fourcc(*'mp4v'),
        fps,
        (w, h),
        True
    )

    if args.bit_depth is not None:
        max_value = 2 ** args.bit_depth
    else:
        max_value = source[0].max_value

    for img_idx in tqdm(indices):
        data_tensor = source[img_idx]
        frame = data_tensor[args.element]

        if DATA_PIXEL_TENSOR_BACKEND == 'torch':
            frame = frame.cpu().numpy()

        if frame.dtype == "uint16":
            frame = frame / max_value
            frame = frame * 255
            frame = frame.astype("uint8")

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if args.downscale:
            frame = cv2.resize(frame, (w, h))

        # --- draw frame index if requested ---
        if args.show_frame_idx:
            text = f"{img_idx}"
            # scale font roughly with frame height, clamped
            font_scale = max(0.5, min(2.0, h / 720.0))
            thickness = 2
            org = (10, int(30 * font_scale + 10))  # left margin, top margin
            # shadow/outline for readability
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        # -------------------------------------

        new_video.write(frame)
        prev_img = frame

    new_video.release()
