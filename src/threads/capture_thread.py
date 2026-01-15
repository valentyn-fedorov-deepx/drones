import threading
import queue
import time
from loguru import logger

from src.camera import CUSTOM_AUTOEXPOSURE_METHODS


class CaptureThread:
    def __init__(self, camera, frame_queue: queue.Queue, stop_event: threading.Event,
                 camera_params_change_queue: queue.Queue):
        self.camera = camera
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.camera_params_change_queue = camera_params_change_queue
        self.latest_data = None

    def capture_loop(self):
        while not self.stop_event.is_set():
            data_tensor = next(self.camera)
            if data_tensor is not None:
                self.latest_data = data_tensor.copy()
                try:
                    self.frame_queue.put_nowait(data_tensor)

                    logger.debug(f"Acquired frame, queue size: {self.frame_queue.qsize()}")

                except queue.Full:
                    logger.debug("Queue is full. Saving cannot keep up! Frame might be dropped.")
                    time.sleep(0.0005)  # Small sleep to prevent busy-waiting on full queue
                    self.frame_queue.get_nowait()

            if not self.camera_params_change_queue.empty():
                set_params_req = self.camera_params_change_queue.get_nowait()
                logger.info(f"Processing camera params change: {set_params_req}")
                if set_params_req.exposure_mode:
                    logger.info(f"Seting exposure mode: {set_params_req.exposure_mode}")
                    self.camera.set_exposure_mode(set_params_req.exposure_mode)

                if set_params_req.exposure and (set_params_req.exposure_mode == self.camera.autoexposure_off_name):
                    logger.info(f"Seting exposure to: {set_params_req.exposure}")
                    self.camera.set_exposure(set_params_req.exposure)

                if set_params_req.gain:
                    logger.info(f"Seting gain: {set_params_req.gain}")
                    self.camera.set_gain(set_params_req.gain)

                if set_params_req.pixel_format:
                    logger.info(f"Seting exposure mode: {set_params_req.exposure_mode}")
                    self.camera.set_pixel_format(set_params_req.pixel_format)

                if set_params_req.width:
                    logger.info(f"Seting width to {set_params_req.width}")
                    self.camera.set_width(set_params_req.width)

                if set_params_req.height:
                    logger.info(f"Seting height to {set_params_req.height}")
                    self.camera.set_height(set_params_req.height)

                if set_params_req.custom_exposure_mode:
                    logger.info(f"Seting custom exposure mode: {set_params_req.custom_exposure_mode}")
                    self.camera.set_custom_autoexposure(CUSTOM_AUTOEXPOSURE_METHODS[set_params_req.custom_exposure_mode])
                else:
                    self.camera.turn_off_custom_autoexposure()

if __name__ == "__main__":
    pass
