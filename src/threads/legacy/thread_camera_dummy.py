import datetime
import logging
import os

from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
import cv2
import numpy as np
import time

from src.gui_config import Config

from .synth_obstacles import synth_obstacle
from src.offline_utils import frame_source
from src.data_pixel_tensor import DataPixelTensor


class CameraDummy264Thread(QThread):
    change_pixmap_signal: pyqtSignal = pyqtSignal(DataPixelTensor)

    def __init__(self, video_path: str, video_pos: int, crop_fov: bool = False,
                 auto_exposure: str = 'OFF', ) -> None:
        self.source = frame_source.FrameSource(video_path, video_pos, None)
        super().__init__()

        self.source.files_mode

        logging.debug(f"{self} __init__")

        self.crop_fov = crop_fov
        self._auto_exposure = auto_exposure
        
        self.obstacle = None
        self.obstacle_coords = (0, 0)

        self.roi = [0,0,0,0]
        
        self._run_flag: bool = False
        self._exposure_value: int = Config.cam_default_exposure
        self._gain_value: int = Config.cam_default_gain
        self._frame = None
        self.fps = 36.6

        self._tm = None
        self._tmm = None

    def run(self):
        try:
            logging.debug(f"{self} run")
            self._run_flag: bool = True

            for data_tensor in self.source:
                frame = (data_tensor.view_img, data_tensor.n_z,
                          data_tensor.n_xy, data_tensor.n_xyz, 0)
                if not self._run_flag:
                    break
                logging.debug(f"{self} while running")
                
                if self._tmm is None:
                    self._tmm = time.time()
                if self._tm is None:
                    dt = 0
                    self._tm = time.time()
                else:
                    dt = time.time() - self._tm
                    self._tm = time.time()
                    
                if self.crop_fov:
                    # Should we move it to the source as an optimization?
                    frame = tuple(img[512:-512, 612:-612] for img in frame)
                    

                if self.obstacle is not None:
                    frame = self._set_overlay_img(frame)

                # Draw roi
                cv2.rectangle(frame[0], (self.roi[0],self.roi[1]), (self.roi[2],self.roi[3]), (255,0,0), 2)
                
                frame_for_hist = cv2.cvtColor(frame[0], cv2.COLOR_BGR2GRAY)
                
                frame_hist = cv2.calcHist([frame_for_hist],[0],None,[256],[0,256])
                frame_hist = frame_hist / frame_hist.sum()
                
                underexposed = frame_hist[:50].sum() 
                overexposed = frame_hist[200:].sum() 
                
                frame_brightness = np.mean(frame_for_hist)

                print(f"NOW: {underexposed}, {overexposed}, {frame_brightness}")
                try:
                    if underexposed or frame_brightness < 50:
                        print(f"Increasing exposure: {underexposed}, {overexposed}, {frame_brightness}")
                        
                    elif overexposed or frame_brightness > 65:
                        print(f"Decreasing exposure: {underexposed}, {overexposed}, {frame_brightness}")    
                        
                except AttributeError as e:
                    print("Error accessing exposure time:", e)
                
                self.change_pixmap_signal.emit(data_tensor)
                self._frame = frame[0]
                self.msleep(int(1000 / self.fps))

                logging.debug(f"{self} image emitted")

            logging.debug(f"{self} capture released")

        except Exception as e:
            logging.exception('General error while running camera thread\n', type(e), e)

    def stop(self):
        logging.debug(f"{self} stop")
        self._run_flag: bool = False
        self.wait()

    def take_photo(self, user_input: str = ""):
        numpy_image: np.ndarray = self._frame
        logging.info(f"{self} taking photo")
        img_name: str = f"{datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S-%f')[:-3]}"
        img_name += f"_{user_input}"
        img_name += f"_exp{self._exposure_value}"
        img_name += f"_gain{self._gain_value}"
        img_name += ".png"
        cv2.imwrite(os.path.join(Config.photo_path, img_name), numpy_image)

    def _set_overlay_img(self, frame):
        pos_x, pos_y = self.obstacle_coords
        # Add obstacle to the frame (modifies numpy arrays in-place)
        self.obstacle.add_obstacle(frame, pos_x, pos_y)
        return frame

    @pyqtSlot(str)
    def set_auto_exposure(self, auto_exposure_value: str) -> None:
        logging.debug(f"auto exposure set to {auto_exposure_value}")
    
    @pyqtSlot(float)
    def set_exposure(self, exposure_value: float):
        logging.debug(f"exposure set to {exposure_value}")

    @pyqtSlot(float)
    def set_gain(self, gain_value: float):
        print(f"gain set to {gain_value}")
        logging.debug(f"gain set to {gain_value}")

    @pyqtSlot(str)
    def set_overlay_img(self, img_name: str):
        # Load an obstacle into the RAM
        if img_name.strip() == '':
            self.obstacle = None
        else:
            self.obstacle = synth_obstacle.SynthObstacle(img_name)
            logging.debug(f"img_name set to {img_name}")

    @pyqtSlot(tuple)
    def set_overlay_coords(self, img_coords: tuple):
        self.obstacle_coords = img_coords
        logging.debug(f"img_name set to {img_coords}")
        
    @pyqtSlot(float)
    def set_nz_scale(self, nz_scale: float):
        nproc = self.source.nproc
        if nproc is not None:
            nproc.scale_factor = nz_scale

    @pyqtSlot(list)
    def set_roi(self, roi: list):
        self.roi = roi
        logging.debug(f"exposure set to {roi}")