from pathlib import Path
import queue
import time
import threading
import shutil
from pathlib import Path
from argparse import ArgumentParser

import cv2
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from loguru import logger
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription

from src.camera import LiveCamera
from deployment.remote_camera_management.request_models import StartRecordingRequest, ResumeRecordingRequest, SetCameraParamsRequest
from deployment.networking_utils import get_local_ip
from src.threads.raw_recorder import DataRecorder
from src.threads.capture_thread import CaptureThread
from src.data_pixel_tensor import DataPixelTensor
from src.camera import CUSTOM_AUTOEXPOSURE_METHODS
from deployment.webrtc_streaming.video_track import NumpyVideoTrack
from deployment.remote_camera_management.stream_page import webrtc_page
from configs.data_pixel_tensor import ELEMENTS_NAMES_WITH_RAW, DATA_PIXEL_TENSOR_BACKEND


class CameraServer:
    def __init__(self, save_path, device_idx=0, as_standalone=True):
        # Initialize camera
        self.live_camera = LiveCamera(device_idx, lazy_calculations=True,
                                      return_data_tensor=True)

        self.save_path = Path(save_path)

        self.recording = False
        self.recording_thread = None
        # self.recording_delay = None
        self.record_name = None

        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_capturing_event = threading.Event()
        self.camera_params_change_queue = queue.Queue(maxsize=5)
        self.data_recorder = None
        self.pcs = set()

        if as_standalone:
            self.app = FastAPI()
            self.app.add_api_route("/capture", self.capture, methods=["GET"])

            self.app.add_api_route("/set_camera_params",
                                self.set_camera_params,
                                methods=["POST"])

            self.app.add_api_route("/start_recording", self.start_recording,
                                methods=["POST"])

            self.app.add_api_route("/stop_recording", self.stop_recording,
                                methods=["POST"])

            self.app.add_api_route("/get_camera_info", self.get_camera_info,
                                methods=["GET"])

            self.app.add_api_route("/get_saved_count", self.get_saved_count,
                                methods=["GET"])

            self.app.add_api_route("/health", self.get_health_check,
                                methods=["GET"])

            self.app.add_api_route("/webrtc", webrtc_page, methods=["GET"])
            self.app.add_api_route("/webrtc/offer", self.webrtc_offer, methods=["POST"])

            local_ip_address = get_local_ip()
            logger.info(f"Local IP address of the device: {local_ip_address}")

        self.capture_thread_processor = CaptureThread(self.live_camera,
                                                      self.frame_queue,
                                                      self.stop_capturing_event,
                                                      self.camera_params_change_queue)
        self.capture_thread = threading.Thread(target=self.capture_thread_processor.capture_loop,
                                               daemon=True)
        self.capture_thread.start()
    
    async def _cleanup_pc(self, pc: RTCPeerConnection):
        try:
            await pc.close()
        finally:
            self.pcs.discard(pc)

    async def _build_peer_connection(self) -> RTCPeerConnection:
        pc = RTCPeerConnection()  # for LAN use; add iceServers here if you need NAT traversal
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            state = pc.connectionState
            logger.info(f"PeerConnection state: {state}")
            if state in ("failed", "closed", "disconnected"):
                await self._cleanup_pc(pc)

        track = NumpyVideoTrack(self.capture_thread_processor, fps=30)

        pc.addTrack(track)
        return pc

    async def webrtc_offer(self, request: Request):
        """
        Receives an SDP offer from the browser, returns an SDP answer.
        """
        try:
            params = await request.json()
            offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid SDP offer: {e}")

        pc = await self._build_peer_connection()

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return JSONResponse(
            content={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        )

    def get_health_check(self):
        return JSONResponse(content={"status": "ok"})

    def get_saved_count(self):
        if not self.recording or self.data_recorder is None:
            rec_number = 0
        else:
            rec_number = self.data_recorder.current_save_idx
        
        if not Path(self.save_path).exists():
            total, used, free = 0, 0, 0 
        else:
            total, used, free = shutil.disk_usage(self.save_path)

        free_gb = round(free / (1024**3), 2)

        return JSONResponse(content=dict(n_records=rec_number,
                                         free_gb=free_gb,
                                         estimated_fps=int(self.live_camera.estimate_fps())))

    def get_camera_info(self):
        camera_info = dict(exposure_range=(self.live_camera.min_exposure + 1, self.live_camera.max_exposure - 1),
                           gain_range=(self.live_camera.min_gain + 1, self.live_camera.max_gain - 1),
                           
                           width_range=(self.live_camera.min_width, self.live_camera.max_width),
                           height_range=(self.live_camera.min_height, self.live_camera.max_height),
                           
                           available_pixel_format=list(self.live_camera.available_pixel_formats.keys()),
                           available_exposure_modes=list(self.live_camera.available_exposure_modes.keys()),
                           available_custom_autoexposure_options=list(CUSTOM_AUTOEXPOSURE_METHODS.keys()),
                           data_elements=ELEMENTS_NAMES_WITH_RAW)

        return JSONResponse(content=camera_info)

    def set_camera_params(self, set_params_req: SetCameraParamsRequest):
        try:
            logger.info(f"Changing camera parameters to {set_params_req}")
            self.camera_params_change_queue.put_nowait(set_params_req)
        except queue.Full:
            time.sleep(0.0005)
            logger.error("Camera params queue is full")
            raise HTTPException("Could not send information to the camera thread. Queue is full.")

        logger.info("Changed camera params")
        return JSONResponse(content={"message": "Changed camera params"})

    def capture(self, element: str = Query("raw", description="Specify which reesentation from the DataPixelTensor should be sent")):
        """Capture a single image and return it as a JPEG stream. """
        try:
            data_tensor = self.capture_thread_processor.latest_data
            if data_tensor is None:
                logger.error(f"Data tensor is None")
                raise ValueError
        except Exception as err:
            logger.error(f"Failed to capture image. {err}")
            raise HTTPException(status_code=500, detail=f"Failed to capture image. {err}")

        try:
            if isinstance(data_tensor, np.ndarray):
                if self.live_camera.packed_pixel_format:
                    data_tensor = DataPixelTensor(data_tensor, width=self.live_camera.width, height=self.live_camera.height,
                                                  color_data=self.live_camera.color_data, lazy_calculations=self.live_camera.lazy_calculations,
                                                  bit_depth=self.live_camera.bit_depth, unpack_method=1)
                else:
                    data_tensor = DataPixelTensor(data_tensor, color_data=self.live_camera.color_data,
                                                  lazy_calculations=self.live_camera.lazy_calculations,
                                                  bit_depth=self.live_camera.bit_depth)

            data_representation = data_tensor[element]
        except ValueError as err:
            logger.error(f"Incorrect element name: {element}. {err}")
            raise HTTPException(status_code=500, detail=f"Incorrect element name: {element}. {err}")

        if DATA_PIXEL_TENSOR_BACKEND == 'torch':
            data_representation = data_representation.cpu().numpy()

        if data_representation.dtype == "uint16":
            # data_representation = data_representation.astype(float) / np.iinfo(np.uint16()).max
            data_representation = data_representation.astype(float) / data_tensor.max_value
            data_representation = np.uint8(data_representation * 255)

        try:
            if len(data_representation.shape) == 2:
                data_representation = cv2.cvtColor(data_representation, cv2.COLOR_GRAY2RGB)

            # new_shape = (data_representation.shape[1] // 5, data_representation.shape[0] // 5)
            # data_representation = cv2.resize(data_representation, new_shape)

            data_representation = cv2.cvtColor(data_representation, cv2.COLOR_RGB2BGR)
            ret, jpeg = cv2.imencode('.jpg', data_representation)
            if not ret:
                logger.error("Failed to encode image")
                raise HTTPException(status_code=500, detail="Failed to encode image")
        except Exception as err:
            logger.error(f"Failed to prepare image for sending: {err}")
            raise HTTPException(status_code=500, detail="Failed to prepare image for sending")

        return StreamingResponse(iter([jpeg.tobytes()]), media_type="image/jpeg")

    def start_recording(self, rec_request: StartRecordingRequest):
        """
        Start video recording in a background thread.
        The request body should include:
          - delay: Delay (in seconds) between frames.
          - record_name: The name for the saved video file.
        """
        if self.recording:
            return JSONResponse(content={"message": "Recording already in progress"})

        record_path = self.save_path / rec_request.record_name
        record_path.mkdir(parents=True, exist_ok=True)
        self.stop_saving_event = threading.Event()

        self.data_recorder = DataRecorder(record_path, self.frame_queue,
                                          self.stop_saving_event)
        if isinstance(rec_request, ResumeRecordingRequest):
            self.data_recorder.current_save_idx = rec_request.begin_frame_idx

        self.recording = True
        self.recording_thread = threading.Thread(target=self.data_recorder.start_recording_loop,
                                                 args=(".pxi", ))
        self.recording_thread.start()
        return JSONResponse(content={"message": "Recording started",
                                     #  "delay": self.recording_delay,
                                     "record_name": str(self.record_name)})

    def stop_recording(self):
        """Stop video recording and wait for the thread to finish."""
        if not self.recording:
            return JSONResponse(content={"message": "Recording is not active"})
        self.recording = False
        self.stop_saving_event.set()
        time.sleep(2)
        self.recording_thread.join()
        self.data_recorder = None
        return JSONResponse(content={"message": "Recording stopped"})

    def set_save_path(self, path):
        self.save_path = Path(path)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--device-idx", default=0, type=int)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--log-level", default="DEBUG")

    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn
    import sys

    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    camera_server = CameraServer(args.save_path, args.device_idx)
    uvicorn.run(camera_server.app, host="0.0.0.0", port=args.port)
