import datetime
import logging
import os

from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
import cv2
import numpy as np
import time

from src.offline_utils import frame_source


class Camera264Thread(QThread):
    change_pixmap_signal: pyqtSignal = pyqtSignal(object)

    def __init__(self, video_path: str) -> None:
        super().__init__()

        logging.debug(f"{self} __init__")
        
        self._run_flag: bool = False
        self._frame = None
        self.fps = 36.6

        self._tm = None
        self._tmm = None

        self.source = frame_source.FrameSource(video_path)

    def run(self):
        try:
            logging.debug(f"{self} run")
            self._run_flag: bool = True

            for frame in self.source: 
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

                self.change_pixmap_signal.emit(frame)
                self._frame = frame
                self.msleep(int(1000 / self.fps))

                logging.debug(f"{self} image emitted")

            logging.debug(f"{self} capture released")

        except Exception as e:
            logging.exception('General error while running camera thread\n', type(e), e)

    def stop(self):
        logging.debug(f"{self} stop")
        self._run_flag: bool = False
        self.wait()

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




        


