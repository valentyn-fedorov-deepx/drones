from datetime import datetime
from logging import debug, exception, info
import os
import time
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal, QMutex, pyqtSlot
import cv2
import numpy as np

from src.gui_config import Config
# from ocr.manager import ProcessingManager
from src.project_managers.project_la_manager import ProjectLAManager

from src.offline_utils import obstacle_cache


class ProcessingThread(QThread):
    batch_yield_signal: pyqtSignal = pyqtSignal(np.ndarray)
    close_message_box_signal: pyqtSignal = pyqtSignal(bool)

    def __init__(self, crop_fov: bool, obstacles_off: bool, use_obstacle_cache: bool = False, video_path = None, roi = None, save_dir = None, dist_refs = [0,0]):
        super().__init__()
        self.use_obstacle_cache = use_obstacle_cache and video_path is not None and not crop_fov
        self.video_path = video_path
        debug(f"{self} __init__")

        self.save_dir = Path(save_dir)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.main_folder_path = self.save_dir / current_time

        # self.processor = ProjectLAManager(crop_fov=crop_fov, obstacles_off=obstacles_off, roi=roi,
        #                                   save_dir=self.main_folder_path, dist_refs=dist_refs)
        self.processor = ProjectLAManager(config_path="configs/project_la/manager.yaml", 
                                          device='cuda')
        self.mutex: QMutex = QMutex()

        self._run_flag: bool = False
        self._idx = 0

        self._frame = None
        self._annotated_frame = None

        self._tms = {}
        self._tm = []
        self._window = 10

    def run(self) -> None:
        try:
            self.mutex.lock()
            self._run_flag: bool = True
            self.mutex.unlock()

            # os.makedirs(self.main_folder_path)
            # image_folder_path = self.main_folder_path / "images"
            # info_folder_path = self.main_folder_path / "info"

            # os.makedirs(image_folder_path)
            # os.makedirs(info_folder_path)

            while self._frame is None:
                self.msleep(10)

            debug(f"{self} run")

            # Check the static obstacle cache or run DINO+SAM
            static_obst = None
            if self.use_obstacle_cache:
                static_obst = obstacle_cache.check_cache(self.video_path)

            if static_obst is None:
                # No cache, we have to run DINO+SAM
                t1 = time.time()
                debug('Initializing: running GroundingDINO + SAM, please wait ...')
                # self.processor.init_processing(self._frame[0])
                debug(f'Initialization finished ({time.time() - t1} s)')
                if self.use_obstacle_cache:
                    obstacle_cache.save_obstacles(self.video_path, self.processor.static_obstacles)
            else:
                debug('Static obstacles loaded successfully from the cache')
                self.processor.static_obstacles = static_obst

            self.close_message_box_signal.emit(True)
            last_processed_frame_ts = 0
            while self._run_flag:
                debug(f"{self} while running")

                t1 = time.time()
                self._idx += 1
                self.mutex.lock()
                # There are multithread issues without this copy

                # for i in self._frame[:4]:
                #     print(i)
                #     print(i.shape)

                # frame = list(f.copy() for f in self._frame[:4])
                # frame.append(self._frame[4])
                # frame.append(self._frame[5])

                self.mutex.unlock()

                # Skip duplicating frame if need
                if last_processed_frame_ts == self._frame.created_at:
                    self.msleep(16)
                    continue
                last_processed_frame_ts = self._frame.created_at

                self.processor.process(self._frame)
                self._annotated_frame = self.processor.generate_vizualization_for_latest_data()
                self.batch_yield_signal.emit(self._annotated_frame)

                dct = dict()
                for k, v in dct.items():
                    self._tms[k] = self._tms.get(k, 0) + v
                for k, v in self._tms.items():
                    print(f"AVG {k} time: {v/self._idx:.3f}s")

                self._tm.append(time.time() - t1)
                if len(self._tm) > self._window:
                    self._tm = self._tm[1:]

                print(f"PROCESSING THREAD FPS: {self._window/sum(self._tm)}")

        except Exception as e:
            exception('General error while running processing thread\n', type(e), e)

    def stop(self) -> None:
        debug(f"{self} stop")
        self._run_flag: bool = False
        self.wait()

    def take_photo(self, user_input: str = "") -> None:
        info(f"{self} taking photo")
        img_name: str = f"{datetime.now().strftime('%Y-%m-%d__%H-%M-%S-%f')[:-3]}"
        img_name += f"_{user_input}"
        img_name += f"_processed"
        img_name += ".png"
        cv2.imwrite(os.path.join(Config.photo_path, img_name), self._annotated_frame)

    @pyqtSlot(tuple)
    def update_image_for_processing(self, cv_img: tuple) -> None:
        self.mutex.lock()
        if self._run_flag:
            self._frame = cv_img
        self.mutex.unlock()
        debug(f"Updated image for processing")

    @pyqtSlot(float)
    def set_range_scale(self, value):
        self.processor.distance_scale_factor = value
