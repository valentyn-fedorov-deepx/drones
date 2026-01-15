from queue import Queue
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, pyqtSlot
import numpy as np
import cv2
import os
from datetime import datetime


class VideoWriterThread(QThread):

    def __init__(self, path, crop_fov=False, limit=50000) -> None:
        super().__init__()

        self._run_flag: bool = False
        self._queue = Queue()

        os.makedirs(path, exist_ok=True)
        self.fps = 4.0

        if crop_fov:
            self.imsize = (1024, 1224)
        else:
            self.imsize = (2048, 2448)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
                os.path.join(path, f"{datetime.now().strftime('%Y-%m-%d__%H-%M-%S-%f')[:-3]}.mp4"), 
                self.fourcc, self.fps, (self.imsize[1], self.imsize[0])
        )

        self.limit = limit
        self._idx = 0

    def run(self) -> None:
        self._run_flag: bool = True
        while self._run_flag:
            try:
                visual = self._queue.get(timeout=1)
                if self._idx > self.limit:
                    self._run_flag = False
                    self.video_writer.release()
                    self.video_writer = None
                else:
                    self._idx += 1
                    self.video_writer.write(visual)
            except:
                pass

    def stop(self) -> None:
        self._run_flag: bool = False
        self.video_writer.release()
        self.wait()

    # slots
    @pyqtSlot(np.ndarray)
    def update_frame_queue(self, cv_img) -> None:
        if self._run_flag:
            self._queue.put(cv_img)
