import cv2
import numpy as np

from deployment.webrtc_streaming.video_track import NumpyVideoTrack


VIDEO_PATH = "/home/oleksii/Documents/2025-07-24 14-20-32_ShapeOS_nxyz.mp4"
FPS = 30


class StubNpCamera(NumpyVideoTrack):
    def __init__(self, video_path=VIDEO_PATH):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        succes, frame = self.cap.read()
        if not succes:
            raise IOError(f"Cannot read video: {video_path}")
        h, w, _ = frame.shape
        super().__init__(w, h, FPS)

    def __del__(self):
        self.cap.release()

    def __next__(self):
        succes, frame = self.cap.read()
        if not succes:
            return None
        return frame

