from queue import Queue
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, pyqtSlot
import numpy as np


class FrameQueueThread(QThread):
    change_pixmap_signal: pyqtSignal = pyqtSignal(np.ndarray)

    def __init__(self) -> None:
        super().__init__()

        self._run_flag: bool = False
        self._queue = Queue()
        self.mutex: QMutex = QMutex()
        self._n = 0

    def run(self) -> None:
        self._run_flag: bool = True
        while self._run_flag:
            import time
            t = time.time()
            try:
                visual, (dt, tm, tmm) = self._queue.get(timeout=1)
                self._n += 1

                delta = time.time() - tm
                # if delta < 0.2:
                #     # self.msleep(int(1000*dt))
                #     self.msleep(int(1000*(tm-tmm)/self._n))
                delta = delta / 0.5
                sleep = 1 if delta < 1 else 1/delta
                self.msleep(int(1000*(tm-tmm)*sleep/self._n))
                
                self.change_pixmap_signal.emit(visual)

                # self.msleep(int(1000/36.6))

                print("TIME_1", time.time() - t, delta, dt)
                t = time.time()
            except:
                pass

    def stop(self) -> None:
        self._run_flag: bool = False
        self.wait()

    # slots
    @pyqtSlot(list)
    def update_queue_batch(self, cv_img_lst) -> None:
        for cv_img in cv_img_lst:
            self._queue.put(cv_img)
