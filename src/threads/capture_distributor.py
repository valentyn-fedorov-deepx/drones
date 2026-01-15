from typing import Optional

from src.threads.capture_thread import CaptureThread

class CaptureThreadDistributor:
    def __init__(self, capture_thread: CaptureThread = None):
        self.capture_thread = capture_thread
        self.consumers = dict()

    @property
    def capture_frame_queue(self):
        return self.capture_thread.frame_queue

    def add_consumer(self, name: str, queue):
        self.consumers[name] = queue

    def remove_consumer(self, name):
        pass

