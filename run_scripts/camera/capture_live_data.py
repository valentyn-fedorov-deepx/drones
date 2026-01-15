from argparse import ArgumentParser
from pathlib import Path
import queue
import threading
import time
import sys

from loguru import logger

from src.camera import LiveCamera
from src.threads.raw_recorder import DataRecorder
from src.threads.capture_thread import CaptureThread
from src.camera.autoexposure.base_autoexposure import BaseAutoExposure


def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--save-path", type=Path)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--color-data", action='store_true')
    parser.add_argument("--log-level", default="INFO")

    return parser.parse_args()


def start_capture(save_path: Path, color_data: bool,
                  duration: float):
    if not save_path.exists():
        save_path.mkdir(parents=True)
    else:
        logger.warning(f"Can override data in the {save_path}")

    live_camera = LiveCamera(return_data_tensor=True, color_data=color_data)

    live_camera.set_exposure_mode("Off")
    # live_camera.set_exposure(500_000)
    # live_camera.set_exposure("100")
    live_camera.set_custom_autoexposure(BaseAutoExposure)
    live_camera.set_pixel_format("Mono12p")

    new_width = int(live_camera.max_width * 0.5)
    new_height = int(live_camera.max_height * 0.5)
    live_camera.set_width(new_width)
    live_camera.set_height(new_height)

    frame_queue = queue.Queue(maxsize=10)
    stop_capturing_event = threading.Event()
    stop_saving_event = threading.Event()
    camera_params_change_queue = queue.Queue(maxsize=5)

    capture_thread_processor = CaptureThread(live_camera,
                                             frame_queue,
                                             stop_capturing_event,
                                             camera_params_change_queue)
    capture_thread = threading.Thread(target=capture_thread_processor.capture_loop,
                                      daemon=True)

    capture_thread.start()

    data_recorder = DataRecorder(save_path, frame_queue,
                                 stop_saving_event)

    recording_thread = threading.Thread(target=data_recorder.start_recording_loop,
                                        args=(".pxi", ))
    recording_thread.start()
    time.sleep(duration)

    stop_capturing_event.set()
    stop_saving_event.set()

    capture_thread.join()
    recording_thread.join()

if __name__ == "__main__":
    args = parse_args()
    
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    start_capture(args.save_path, args.color_data,
                  args.duration)

    captured_files = list(args.save_path.glob("*"))
    total_files = len(captured_files)
    estimated_fps = total_files / args.duration
    logger.info(f"Captured {total_files} files, estimated FPS is {estimated_fps:.2f}")