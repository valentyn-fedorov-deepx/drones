from datetime import datetime, time
import logging
from logging import error, debug, info, exception
from os import path
import time as timemodule
import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
import cv2

from src.data_pixel_tensor import DataPixelTensor

from src.camera.gxipy import (
    DeviceManager, U3VDevice, U2Device, GEVDevice, RawImage, GxSwitchEntry, GxFrameStatusList, DataStream, GxAutoEntry)
from src.camera.daheng_camera import DahengCamera, get_numpy_image_from_stream

from .synth_obstacles import synth_obstacle
from src.gui_config import Config


class Camera264Thread(QThread):
    change_pixmap_signal: pyqtSignal = pyqtSignal(DataPixelTensor)

    def __init__(self, crop_fov=False, auto_exposure: str = 'OFF') -> None:
        super().__init__()
        # placeholders
        self._camera_capture = None
        self.obstacle = None
        self.obstacle_coords = (0, 0)

        self.roi = [0,0,0,0]
        
        self._crop_fov = crop_fov
        self._auto_exposure = auto_exposure
        
        # self._preprocessor = NormalsProcessorCUDA()

        self._window = 10
        self._tm = []
        
        self.evening_start = time(16, 51)  
        self.morning_end = time(6, 48)    

        debug(f"{self} __init__")

        self._run_flag: bool = False
        self._exposure_value: int = Config.cam_default_exposure
        self._gain_value: int = Config.cam_default_gain

    def run(self) -> None:
        try:
            debug(f"{self} run")
            self._run_flag: bool = True
            self._camera_capture, self._device_manager = DahengCamera.setup_camera()

            while self._run_flag:
                debug(f"{self} while running")

                t1 = timemodule.time()
                frame = get_numpy_image_from_stream(self._camera_capture.data_stream[0])
                if self._crop_fov:
                    frame = frame[512:-512, 612:-612]
                data_tensor = DataPixelTensor(frame, color_data=False, for_display=True)
                data_tensor.calculate_all()
                data_tensor.convert_to_numpy()
                # u_i, n_z, n_xy, n_xyz = self._preprocessor.get_normals(frame, ui=True, nz=True, nxy=True, nxyz=True)

                # frame = (u_i, n_z, n_xy, n_xyz)
                frame = (data_tensor.view_img, data_tensor.n_z, data_tensor.n_xy, data_tensor.n_xyz)
                if self.obstacle is not None:
                    frame = self._set_overlay_img(frame)
                self._tm.append(timemodule.time() - t1)
                if len(self._tm) > self._window:
                    self._tm = self._tm[1:]
                tm_velocity = timemodule.time()
                frame = frame + (tm_velocity,)
                # frame = frame + (normals_frame,)
                
                print(f"CAMERA THREAD FPS: {self._window / sum(self._tm)}")

                # Draw ROI
                cv2.rectangle(frame[0], (self.roi[0],self.roi[1]), (self.roi[2],self.roi[3]), (255,0,0), 1)
                
                current_exposure_auto = self._camera_capture.ExposureAuto.get()
                
                # Get frame HIST
                if self._auto_exposure == 'CUSTOM':
                        
                    current_time = datetime.now().time()
                    
                    frame_for_hist = cv2.cvtColor(frame[0], cv2.COLOR_BGR2GRAY)
                    
                    frame_hist = cv2.calcHist([frame_for_hist],[0],None,[256],[0,256])
                    frame_hist = frame_hist / frame_hist.sum()
                    
                    underexposed = frame_hist[:50].sum() > 0.6
                    overexposed = frame_hist[200:].sum() > 0.6
                    
                    frame_brightness = np.mean(frame_for_hist)
                    
                    if current_time >= self.evening_start or current_time <= self.morning_end:
                        self._camera_capture.ExposureAuto.set(GxAutoEntry.OFF)
                        
                        self._camera_capture.Gain.set(20)
                        self._camera_capture.ExposureTime.set(200000)
                    
                    elif current_time >= self.morning_end and current_time <= time(7, 48):
                        if current_exposure_auto != GxAutoEntry.CONTINUOUS:
                            self._camera_capture.ExposureAuto.set(GxAutoEntry.CONTINUOUS)
                        
                    elif current_time >= self.evening_start and current_time <= time(17, 51):
                        if current_exposure_auto != GxAutoEntry.CONTINUOUS:
                            self._camera_capture.ExposureAuto.set(GxAutoEntry.CONTINUOUS)
                        
                    else:
                        self._camera_capture.ExposureAuto.set(GxAutoEntry.OFF)
                        
                        self._camera_capture.Gain.set(0)

                        try:
                            exposure_time = self._camera_capture.ExposureTime.get()  
                            print(frame_hist[:50].sum(), frame_hist[200:].sum(), frame_brightness)
                            
                            if underexposed or frame_brightness < 60:
                                n_expo = exposure_time + 10
                                print(f"Increasing exposure: {n_expo}")
                                
                                self._camera_capture.ExposureTime.set(n_expo)
                            elif overexposed or frame_brightness > 80:
                                n_expo = exposure_time - 10
                                print(f"Decreasing exposure: {n_expo}")
                                
                                self._camera_capture.ExposureTime.set(n_expo)
                                
                        except AttributeError as e:
                            print("Error accessing exposure time:", e)
                
                self.change_pixmap_signal.emit(data_tensor)
                debug(f"{self} image emitted")

            # Stop acquisition, close device
            self._camera_capture.stream_off()
            self._camera_capture.close_device()
            debug(f"{self} capture released")

        except Exception as e:
            exception('General error while running camera thread\n', type(e), e)

    def stop(self) -> None:
        debug(f"{self} stop")
        self._run_flag: bool = False
        self.wait()

    def take_photo(self, user_input: str = "") -> None:
        numpy_image = get_numpy_image_from_stream(self._camera_capture.data_stream[0])
        if self._crop_fov:
            numpy_image = numpy_image[512:-512, 612:-612]

        info(f"{self} taking photo")
        img_name: str = f"{datetime.now().strftime('%Y-%m-%d__%H-%M-%S-%f')[:-3]}"
        img_name += f"_{user_input}"
        img_name += f"_exp{self._exposure_value}"
        img_name += f"_gain{self._gain_value}"
        img_name += ".png"
        cv2.imwrite(path.join(Config.photo_path, img_name), numpy_image)

    def _set_overlay_img(self, frame):
        pos_x, pos_y = self.obstacle_coords
        # Add obstacle to the frame (modifies numpy arrays in-place)
        self.obstacle.add_obstacle(frame, pos_x, pos_y)
        return frame
    
    @pyqtSlot(str)
    def set_auto_exposure(self, auto_exposure_value: str) -> None:
        if self._camera_capture is not None:
            if auto_exposure_value == 'OFF':
                self._camera_capture.ExposureAuto.set(GxAutoEntry.OFF)
            elif auto_exposure_value == 'CONTINUOUS':
                self._camera_capture.ExposureAuto.set(GxAutoEntry.CONTINUOUS)
            elif auto_exposure_value == 'ONCE':
                self._camera_capture.ExposureAuto.set(GxAutoEntry.ONCE)
            elif auto_exposure_value == 'CUSTOM':
                self._camera_capture.ExposureAuto.set(GxAutoEntry.OFF)
        self._auto_exposure: str = auto_exposure_value
        debug(f"auto exposure set to {auto_exposure_value}")
    
    @pyqtSlot(float)
    def set_exposure(self, exposure_value: float) -> None:
        if self._camera_capture is not None:
            self._camera_capture.ExposureTime.set(exposure_value)
        self._exposure_value: int = int(exposure_value)
        debug(f"exposure set to {exposure_value}")

    @pyqtSlot(float)
    def set_gain(self, gain_value: float) -> None:
        if self._camera_capture is not None:
            self._camera_capture.Gain.set(gain_value)
        self._gain_value: int = int(gain_value)
        debug(f"gain set to {gain_value}")

    @pyqtSlot(str)
    def set_overlay_img(self, img_name: str) -> None:
        # Load an obstacle into the RAM
        if img_name.strip() == '':
            self.obstacle = None
        else:
            self.obstacle = synth_obstacle.SynthObstacle(img_name)
            logging.debug(f"img_name set to {img_name}")

    @pyqtSlot(tuple)
    def set_overlay_coords(self, img_coords: tuple) -> None:
        print("set_overlay_coords")
        self.obstacle_coords = img_coords
        debug(f"img_name set to {img_coords}")
        
    @pyqtSlot(float)
    def set_nz_scale(self, nz_scale: float):
        pass

    @pyqtSlot(list)
    def set_roi(self, roi: list):
        self.roi = roi
        logging.debug(f"exposure set to {roi}")