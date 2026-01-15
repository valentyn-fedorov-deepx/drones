import time
from functools import wraps
from collections import deque


from loguru import logger
from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension
from ids_peak_ipl import ids_peak_ipl
import numpy as np

from src.data_pixel_tensor import DataPixelTensor
from src.utils.raw_processing.utils_raw_processing_numpy import decode_mono12_contiguous_byteswapped


ids_peak.Library.Initialize()
PIXEL_FORMAT_BIT_DEPTH = {
                        'Mono8': 8,
                        'BayerGR8': 8,
                        'BayerRG8': 8,
                        'BayerGB8': 8,
                        'BayerBG8': 8,
                        'Mono10p': 10,
                        'BayerBG10p': 10,
                        'BayerGB10p': 10,
                        'BayerGR10p': 10,
                        'BayerRG10p': 10,
                        'Mono12p': 12,
                        'BayerBG12p': 12,
                        'BayerGB12p': 12,
                        'BayerGR12p': 12,
                        'BayerRG12p': 12,
                        'Mono10': 10,
                        'Mono12': 12,
                        'BayerGR10': 10,
                        'BayerRG10': 10,
                        'BayerGB10': 10,
                        'BayerBG10': 10,
                        'BayerGR12': 12,
                        'BayerRG12': 12,
                        'BayerGB12': 12,
                        'BayerBG12': 12,
                        'RGB8': 8,
                        'BGR8': 8,
                        }


def apply_while_idle(reannounce_buffers=False, param_getter=None):
    """
    Wraps a method to ensure it's applied only when acquisition is idle.

    If a `param_getter` function is provided, it's used to check if the new
    value is different from the current one. If they are the same, the function
    exits early without interrupting the camera stream.

    Args:
        reannounce_buffers (str): True, or False.
        param_getter (callable, optional): A function (like a lambda) that
            takes the class instance `self` and returns the current value of the
            parameter to be checked. Defaults to None.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            # If a getter is provided, check if the value actually needs to change
            if param_getter:
                current_value = param_getter(self)
                new_value = args[0] if args else None

                if current_value == new_value:
                    # Use the function's name in the log for clarity
                    param_name = fn.__name__.replace('set_', '')
                    logger.info(f"{param_name.capitalize()} is already set to {new_value}. No changes made.")
                    return

            # --- If here, a change is needed or no check was performed ---
            was_streaming = False
            old_payload = None

            try:
                if self.datastream:
                    was_streaming = bool(self.datastream.IsGrabbing())
            except Exception:
                was_streaming = False

            if was_streaming and reannounce_buffers == True:
                old_payload = self.nodemap_remote_device.FindNode("PayloadSize").Value()

            try:
                if was_streaming:
                    self.stop_acquisition()

                result = fn(self, *args, **kwargs)

                if was_streaming and reannounce_buffers == True:
                    new_payload = self.nodemap_remote_device.FindNode("PayloadSize").Value()
                    needs_reannounce = (reannounce_buffers is True) or (old_payload != new_payload)
                    if needs_reannounce and self.datastream:
                        self.reconfigure_buffers()

                return result
            finally:
                if was_streaming:
                    self.start_acquisition()
        return wrapper
    return decorator


def get_available_entries(nodemap_remote, node_name: str):
    try:
        node = nodemap_remote.FindNode(node_name)
        available_entries = {entry.DisplayName(): entry.Value() for entry in node.Entries() if entry.IsAvailable()}
        return available_entries
    except:
        raise ValueError(f"Couldn't get entries for the {node_name} node")


class IdsPeakCamera:
    """
    A class to interact with an IDS peak camera for continuous image acquisition.
    """

    def __init__(self, device_idx: int = 0, lazy_calculations: bool = True,
                 return_data_tensor: bool = True, color_data: bool = True):
        self.lazy_calculations = lazy_calculations
        self.device = None
        self.datastream = None
        self.nodemap_remote_device = None
        self.return_data_tensor = return_data_tensor
        self._custom_autoexposure_active = False
        self.device_idx = device_idx
        self.color_data = color_data
        self.failed_attempts = 0
        # Stats for FPS estimation (based only on frames already yielded by __next__)
        self._frame_times = deque(maxlen=128)  # monotonic timestamps of successful frames
        self._fps_ema = None                   # exponential moving average of instantaneous FPS
        self._fps_window_s = 2.0               # sliding window (seconds) used for estimate_fps()

        try:
            self.init()
        except ids_peak.Exception as e:
            logger.error(f"ERROR initializing camera: {e}")
            self.close()
            raise
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}")
            self.close()
            raise

    @property
    def min_height(self):
        return self.nodemap_remote_device.FindNode("Height").Minimum()

    @property
    def max_height(self):
        return self.nodemap_remote_device.FindNode("Height").Maximum()

    @property
    def min_width(self):
        return self.nodemap_remote_device.FindNode("Width").Minimum()

    @property
    def max_width(self):
        return self.nodemap_remote_device.FindNode("Width").Maximum()

    @property
    def min_exposure(self):
        exp_node = self.nodemap_remote_device.FindNode("ExposureTime")
        min_exposure = exp_node.Minimum()
        return min_exposure

    @property
    def max_exposure(self):
        exp_node = self.nodemap_remote_device.FindNode("ExposureTime")
        max_exposure = exp_node.Maximum()

        return max_exposure

    @property
    def max_gain(self):
        gain_node = self.nodemap_remote_device.FindNode("Gain")
        max_gain = gain_node.Maximum()
        return max_gain

    @property
    def min_gain(self):
        gain_node = self.nodemap_remote_device.FindNode("Gain")
        min_gain = gain_node.Minimum()
        return min_gain

    def init(self):
        logger.info("IDS peak library initialized.")

        dm = ids_peak.DeviceManager.Instance()
        dm.Update()
        devs = dm.Devices()
        if not devs:
            raise RuntimeError("No IDS peak camera found.")
        logger.info(f"Found {len(devs)} device(s).")

        self.device = devs[self.device_idx].OpenDevice(ids_peak.DeviceAccessType_Control)
        logger.info(f"Opened device: {self.device.ModelName()} (S/N: {self.device.SerialNumber()})")

        self.nodemap_remote_device = self.device.RemoteDevice().NodeMaps()[self.device_idx]

        # Basic config
        try:
            acq_mode = self.nodemap_remote_device.FindNode("AcquisitionMode")
            acq_mode.SetCurrentEntry("Continuous")
            logger.info(f"Acquisition mode set to: {acq_mode.CurrentEntry().SymbolicValue()}")
        except ids_peak.Exception as e:
            logger.warning(f"Could not set AcquisitionMode: {e}")

        try:
            trig_sel = self.nodemap_remote_device.FindNode("TriggerSelector")
            trig_sel.SetCurrentEntry("FrameStart")
            trig_mode = self.nodemap_remote_device.FindNode("TriggerMode")
            trig_mode.SetCurrentEntry("Off")
            logger.info("Trigger mode set to Off (Free Run).")
        except ids_peak.Exception as e:
            logger.warning(f"Could not disable trigger mode: {e}")

        dss = self.device.DataStreams()
        if not dss:
            raise RuntimeError("Device has no DataStreams.")
        self.datastream = dss[0].OpenDataStream()
        logger.info("Data stream opened.")

        self.reconfigure_buffers()

        self.start_acquisition()

        if self.color_data is None:
            if "Mono8" in self.available_pixel_formats:
                self.color_data = False
            else:
                self.color_data = True

    def reinit(self):
        self.close()
        time.sleep(2)
        
        # reset fps stats
        self._frame_times.clear()
        self._fps_ema = None
        
        self.init()

    @property
    def acquisition_running(self) -> bool:
        try:
            return bool(self.datastream is not None and self.datastream.IsGrabbing())
        except Exception:
            return False

    def stop_acquisition(self):
        """Stops acquisition safely."""
        logger.info("Stopping acquisition...")
        try:
            # Ensure the datastream is stopped first (important!)
            self.datastream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
            logger.info("  Datastream stopped.")

            # Send AcquisitionStop command to the device
            self.nodemap_remote_device.FindNode("AcquisitionStop").Execute()
            self.nodemap_remote_device.FindNode("AcquisitionStop").WaitUntilDone()  # Wait for command completion
            logger.info("  AcquisitionStop command executed.")

            self.datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
            self.unlock_parameters()
            logger.info("  Datastream flushed.")

        except ids_peak.Exception as e:
            logger.error(f"Error stopping acquisition: {e}")
            # Handle error appropriately

    def start_acquisition(self):
        """Starts acquisition safely."""
        logger.info("Starting acquisition...")
        try:
            # Start the datastream acquisition (prepares it for buffers)
            self.datastream.StartAcquisition()
            logger.info("  Datastream started.")

            self.unlock_parameters()

            # Send AcquisitionStart command to the device
            self.nodemap_remote_device.FindNode("AcquisitionStart").Execute()
            self.nodemap_remote_device.FindNode("AcquisitionStart").WaitUntilDone()  # Wait for command completion
            logger.info("  AcquisitionStart command executed.")

        except ids_peak.Exception as e:
            logger.error(f"Error starting acquisition: {e}")
            # Handle error appropriately

    def reconfigure_buffers(self, additional_buffers: int = 20):
        logger.info("Re-announcing and queueing buffers with new size...")
        payload_size = self.nodemap_remote_device.FindNode("PayloadSize").Value()
        logger.info(f"New PayloadSize: {payload_size} bytes")

        try:
            self.datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)

            if self.datastream.IsGrabbing():
                self.datastream.Close()
                self.datastream = self.device.DataStreams()[0].OpenDataStream()

            buf_count = self.datastream.NumBuffersAnnouncedMinRequired() + additional_buffers
            buffers = []
            for _ in range(buf_count):
                buf = self.datastream.AllocAndAnnounceBuffer(payload_size)
                self.datastream.QueueBuffer(buf)
                buffers.append(buf)
            logger.info(f"  {len(buffers)} buffers allocated and queued.")
        except ids_peak.Exception as e:
            logger.error(f"Error re-allocating buffers: {e}")

    def set_exposure_mode(self, exposure_mode: str):
        if exposure_mode not in self.available_exposure_modes:
            logger.error(f"Exposure mode {exposure_mode} not recognized.")
            return

        if exposure_mode == self.current_exposure:
            logger.error(f"Exposure {exposure_mode} already set.")
            return
    
        exposure_mode = self.available_exposure_modes[exposure_mode]
        node = self.nodemap_remote_device.FindNode("ExposureAuto")
        if node.IsReadable() and node.IsWriteable():
            current = node.CurrentEntry().SymbolicValue()
            logger.info(f"Current ExposureAuto mode: {current}")
            if current != exposure_mode:
                node.SetCurrentEntry(exposure_mode)
                logger.info(f"Set ExposureAuto to: {exposure_mode}")
            else:
                logger.info(f"ExposureAuto already {exposure_mode}.")
        else:
            logger.warning("Cannot access ExposureAuto node.")

    @apply_while_idle(reannounce_buffers=True, param_getter=lambda s: s.width)
    def set_width(self, new_width: int):
        logger.info(f"Applying new width: {new_width}")
        node = self.nodemap_remote_device.FindNode("Width")
        node.SetValue(new_width)

    @apply_while_idle(reannounce_buffers=True, param_getter=lambda s: s.height)
    def set_height(self, new_height: int):
        logger.info(f"Applying new height: {new_height}")
        node = self.nodemap_remote_device.FindNode("Height")
        node.SetValue(new_height)

    @apply_while_idle(reannounce_buffers=True, param_getter=lambda s: s.current_pixel_format)
    def set_pixel_format(self, pixel_format_name: str):
        available = self.available_pixel_formats
        if pixel_format_name not in available:
            raise ValueError(f"Pixel format {pixel_format_name} not supported")

        node = self.nodemap_remote_device.FindNode("PixelFormat")
        if not node.IsWriteable():
            raise RuntimeError("PixelFormat node not writable")

        logger.info(f"Applying new pixel format: {pixel_format_name}")
        node.SetCurrentEntry(pixel_format_name)
        return True

    def set_max_frame_rate(self):
        frame_rate_node = self.nodemap_remote_device.FindNode("AcquisitionFrameRate")
        current_value = frame_rate_node.Value()
        maximum_value = frame_rate_node.Maximum()
        if current_value < (maximum_value - 1):
            frame_rate_node.SetValue(maximum_value)

    def set_exposure(self, exposure: float):
        if self.min_exposure < exposure < self.max_exposure and exposure != self.current_exposure:
            self.nodemap_remote_device.FindNode("ExposureTime").SetValue(exposure)

    def set_gain(self, gain: float):
        if self.min_gain < gain < self.max_gain and gain != self.current_gain:
            self.nodemap_remote_device.FindNode("Gain").SetValue(gain)

    def set_custom_autoexposure(self, autoexposure_cls):
        if autoexposure_cls is None:
            self.turn_off_custom_autoexposure()
            return
        if self._custom_autoexposure_active:
            logger.info("Replacing custom autoexposure method")
        self._custom_autoexposure = autoexposure_cls(max_possible_value=self.max_value,
                                                     max_exposure_value=self.max_exposure * 0.7)
        self._custom_autoexposure_active = True
        self.turn_off_camera_autoexposure()
    
    def turn_off_custom_autoexposure(self):
        logger.info("Turning off custom autoexposure method")
        if self._custom_autoexposure_active:
            self._custom_autoexposure = None
            self._custom_autoexposure_active = False

    def turn_off_camera_autoexposure(self):
        self.set_exposure_mode(self.autoexposure_off_name)

    def __iter__(self):
        return self

    @property
    def autoexposure_off_name(self):
        return "Off"

    @property
    def available_pixel_formats(self):
        return get_available_entries(self.nodemap_remote_device, "PixelFormat")

    @property
    def available_exposure_modes(self):
        return get_available_entries(self.nodemap_remote_device, "ExposureAuto")

    @property
    def bit_depth(self):
        return PIXEL_FORMAT_BIT_DEPTH[self.current_pixel_format]

    @property
    def max_value(self):
        return (2 ** self.bit_depth) - 1

    @property
    def current_gain(self):
        return self.nodemap_remote_device.FindNode("Gain").Value()

    @property
    def current_exposure(self):
        return self.nodemap_remote_device.FindNode("ExposureTime").Value()

    @property
    def current_exposure_mode(self):
        return self.nodemap_remote_device.FindNode("ExposureAuto").CurrentEntry().DisplayName()

    @property
    def current_pixel_format(self):
        return self.nodemap_remote_device.FindNode("PixelFormat").CurrentEntry().DisplayName()

    @property
    def packed_pixel_format(self):
        return self.current_pixel_format.endswith("p")

    @property
    def width(self):
        return self.nodemap_remote_device.FindNode("Width").Value()
    
    @property
    def height(self):
        return self.nodemap_remote_device.FindNode("Height").Value()

    def __next__(self):
        if not self.acquisition_running or not self.datastream:
            logger.warning("Acquisition not running or datastream unavailable.")
            self.close()
            raise StopIteration
        
        if self.failed_attempts > 10:
            logger.warning(f"Couldn't get proper frame after 10 attempts, trying to reinit the camera")
            self.reinit()

        try:
            buf = self.datastream.WaitForFinishedBuffer(3000)
            if buf.HasImage():
                ipl_img = ids_peak_ipl_extension.BufferToImage(buf)

                if self.packed_pixel_format:
                    memory_img = ipl_img.DataView()
                    arr = np.frombuffer(memory_img, dtype=np.uint8).copy()
                else:
                    arr = ipl_img.get_numpy().copy()
                self.datastream.QueueBuffer(buf)

                if arr.size == 0:
                    self.failed_attempts += 1
                    return None

                logger.debug(f"Received array with {arr.shape} dimensions from the Buffer")

                self._note_frame_event()

                self.set_max_frame_rate()
                self.failed_attempts = 0
                if self.return_data_tensor:
                    if self.packed_pixel_format:
                        tensor = DataPixelTensor(arr, width=self.width, height=self.height,
                                                 color_data=self.color_data, lazy_calculations=self.lazy_calculations,
                                                 bit_depth=self.bit_depth, unpack_method=1)
                    else:
                        tensor = DataPixelTensor(arr, color_data=self.color_data,
                                                lazy_calculations=self.lazy_calculations,
                                                bit_depth=self.bit_depth)
                    if self._custom_autoexposure_active:
                        current_exposure = self.current_exposure
                        new_exposure = self._custom_autoexposure(tensor.raw, current_exposure)
                        if new_exposure != current_exposure:
                            self.set_exposure(new_exposure)

                    return tensor
                else:
                    if self._custom_autoexposure_active:
                        if self.packed_pixel_format:
                            tensor = DataPixelTensor(arr, width=self.width, height=self.height,
                                                    color_data=self.color_data, lazy_calculations=self.lazy_calculations,
                                                    bit_depth=self.bit_depth, unpack_method=1)
                        else:
                            tensor = DataPixelTensor(arr, color_data=self.color_data,
                                                    lazy_calculations=self.lazy_calculations,
                                                    bit_depth=self.bit_depth)
                        current_exposure = self.current_exposure
                        new_exposure = self._custom_autoexposure(tensor.raw, current_exposure)
                        if new_exposure != current_exposure:
                            self.set_exposure(new_exposure)
                    return arr
            else:
                self.failed_attempts += 1
                status = buf.Status()
                self.datastream.QueueBuffer(buf)
                logger.error(f"Buffer error. Status: {status.ToString()} ({status.Value()})")
                return None

        except ids_peak.Exception as e:
            logger.error(f"ERROR during acquisition: {e}")
            self.failed_attempts += 1
            return None
        
    def estimate_fps(self):
        """
        Estimate the actual capture FPS using timing of frames already delivered by __next__.

        Strategy:
        - Prefer a sliding-window estimate over the last self._fps_window_s seconds:
                (frames_in_window - 1) / (time_span_in_window)
        - If the window is too sparse, fall back to an exponential moving average (EMA)
            of instantaneous FPS that we update on every frame.
        - As a last resort, use the entire history currently in the deque.

        Returns:
            float | None: Estimated FPS, or None if not enough data yet.
        """
        times = list(self._frame_times)
        if len(times) < 2:
            return -1

        now = time.monotonic()
        cutoff = now - self._fps_window_s
        recent = [t for t in times if t >= cutoff]

        # Windowed estimator
        if len(recent) >= 2:
            span = recent[-1] - recent[0]
            if span > 0:
                return (len(recent) - 1) / span

        # EMA fallback (more stable during sparse windows)
        if self._fps_ema is not None:
            return float(self._fps_ema)

        # Last resort: use the entire deque
        span_all = times[-1] - times[0]
        if span_all > 0:
            return (len(times) - 1) / span_all

        return -1


    def unlock_parameters(self):
        self.nodemap_remote_device.FindNode("TLParamsLocked").SetValue(0)
        logger.info("Parameters unlocked.")

    def lock_parameters(self):
        self.nodemap_remote_device.FindNode("TLParamsLocked").SetValue(1)

    def close(self):
        logger.info("Closing camera...")
        if self.acquisition_running:
            self.stop_acquisition()
        logger.info("Acquisition stopped.")

        if self.nodemap_remote_device:
            try:
                self.unlock_parameters()
            except ids_peak.Exception as e:
                logger.warning(f"Could not unlock parameters: {e}")
            except AttributeError:
                logger.warning("nodemap_remote_device not available for unlock.")

        self.datastream = None
        self.nodemap_remote_device = None
        self.device = None
        logger.info("Camera resources released.")

    def __del__(self):
        self.close()

    def _note_frame_event(self):
        """Record a successful frame arrival time; updates EMA FPS."""
        now = time.monotonic()
        if self._frame_times:
            dt = now - self._frame_times[-1]
            if dt > 0:
                inst = 1.0 / dt
                self._fps_ema = inst if self._fps_ema is None else (0.2 * inst + 0.8 * self._fps_ema)
        self._frame_times.append(now)


if __name__ == "__main__":
    from tqdm import tqdm
    import numpy as np
    import cv2
    camera = IdsPeakCamera(return_data_tensor=False)
    # camera.set_exposure_mode("Continuous")
    camera.set_exposure_mode("Off")
    camera.set_exposure(100)
    camera.set_pixel_format("Mono8")
    
    camera.set_height(camera.max_height)
    camera.set_width(camera.max_width)

    camera.set_height(1000)
    camera.set_width(1000)

    time_history = list()
    fps_history = list()

    max_frames = 40
    for i in range(max_frames):
        start_time = time.monotonic()
        data_tensor = next(camera)
        duration = time.monotonic() - start_time
        fps = 1 / duration
        logger.info(f"Capture: {duration:.3}s, FPS: {int(fps)}")

        time_history.append(duration)
        fps_history.append(fps)


    print(f"mean time: {np.mean(time_history)}, median time: {np.median(time_history)}")
    print(f"mean fps: {np.mean(fps_history)}, median fps: {np.median(fps_history)}")
