import time
from functools import wraps
from typing import Callable, TypeVar, Any

from loguru import logger
try:
    from src.camera.gxipy import GxFrameStatusList, GxSwitchEntry, GxAutoEntry, GxPixelFormatEntry, GxTriggerSourceEntry
    from src.camera.gxipy.gxiapi import DeviceManager

    from src.camera.gxipy.gxwrapper import gx_get_interface_number
except:
    logger.debug("Loading old gxipy version")
    from src.camera.gxipy_old import GxFrameStatusList, GxSwitchEntry, GxAutoEntry, GxPixelFormatEntry, GxTriggerSourceEntry
    from src.camera.gxipy_old.gxiapi import DeviceManager

from src.data_pixel_tensor import DataPixelTensor


# Pixel format to bit depth mapping
PIXEL_FORMAT_BIT_DEPTH = {
    # Monochrome
    GxPixelFormatEntry.MONO8: 8,
    GxPixelFormatEntry.MONO10: 10,
    GxPixelFormatEntry.MONO12: 12,
    GxPixelFormatEntry.MONO16: 16,

    # Bayer (Raw Color)
    GxPixelFormatEntry.BAYER_RG8: 8,
    GxPixelFormatEntry.BAYER_GB8: 8,
    GxPixelFormatEntry.BAYER_GR8: 8,
    GxPixelFormatEntry.BAYER_BG8: 8,
    GxPixelFormatEntry.BAYER_RG10: 10,
    GxPixelFormatEntry.BAYER_GB10: 10,
    GxPixelFormatEntry.BAYER_GR10: 10,
    GxPixelFormatEntry.BAYER_BG10: 10,
    GxPixelFormatEntry.BAYER_RG12: 12,
    GxPixelFormatEntry.BAYER_GB12: 12,
    GxPixelFormatEntry.BAYER_GR12: 12,
    GxPixelFormatEntry.BAYER_BG12: 12,
    GxPixelFormatEntry.BAYER_RG16: 16,
    GxPixelFormatEntry.BAYER_GB16: 16,
    GxPixelFormatEntry.BAYER_GR16: 16,
    GxPixelFormatEntry.BAYER_BG16: 16,
}


def _nearest_valid_size_and_inc(requested: int, *, min_v: int, max_v: int, inc: int) -> int:
    """
    Return the closest value to `requested` that lies in [min_v, max_v]
    and satisfies (value - min_v) % inc == 0. On a tie, prefer the larger value.
    NOTE: This version no longer checks divisibility by 8—only the increment grid.
    """
    inc = int(inc or 1)
    if inc < 1:
        inc = 1

    # Clamp first so we only consider the legal range
    v = int(requested)
    if v < min_v:
        v = min_v
    elif v > max_v:
        v = max_v

    # Find nearest grid points: floor and ceil on the inc-grid starting at min_v
    offset = v - min_v
    k_floor = offset // inc  # integer floor
    cand_floor = min_v + k_floor * inc
    cand_ceil = cand_floor if (offset % inc == 0) else cand_floor + inc

    # Discard candidates outside the range
    if cand_floor < min_v:
        cand_floor = None
    if cand_ceil > max_v:
        cand_ceil = None

    # Choose the best available candidate
    if cand_floor is None and cand_ceil is None:
        # Degenerate case: no grid point falls in range (e.g., min_v > max_v).
        # Fallback to the clamped value.
        return v
    if cand_floor is None:
        return int(cand_ceil)
    if cand_ceil is None:
        return int(cand_floor)

    # Both valid: pick the closest; break ties upward
    diff_floor = abs(v - cand_floor)
    diff_ceil = abs(cand_ceil - v)
    if diff_floor < diff_ceil:
        return int(cand_floor)
    if diff_ceil < diff_floor:
        return int(cand_ceil)
    return int(max(cand_floor, cand_ceil))


T = TypeVar("T")

def restart_acquisition_if_running(fn: Callable[..., T]) -> Callable[..., T]:
    """Decorator: if acquisition is running, stop -> call -> restart."""
    @wraps(fn)
    def wrapper(self, *args: Any, **kwargs: Any) -> T:
        was_running = bool(getattr(self, "acquisition_running", False))
        if was_running:
            logger.debug(f"Stopping acquisition before {fn.__name__}")
            self.stop_acquisition()
        try:
            return fn(self, *args, **kwargs)
        finally:
            if was_running:
                logger.debug(f"Restarting acquisition after {fn.__name__}")
                self.start_acquisition()
    return wrapper  # type: ignore[return-value]


def get_numpy_image_from_stream(stream, current_exposure_value_ns=1_000_000_000,
                                not_ret_frames: int = 30):
    not_ret: int = 0
    exposure_sec = current_exposure_value_ns * 1E-09

    # Calculate a reasonable timeout based on exposure time
    # For high exposures, we need a proportional but reasonable wait time
    wait_time = min(1.0, exposure_sec * 2) + 0.5  # Scale with exposure but cap reasonably

    while not_ret <= not_ret_frames:
        # Get an image from the 0th stream channel
        raw_image = stream.get_image()

        if raw_image is None or raw_image.get_status() == GxFrameStatusList.INCOMPLETE:
            logger.warning(f"no raw image, attempt {not_ret+1}/{not_ret_frames+1}")
            time.sleep(wait_time)
            not_ret += 1
            if not_ret > not_ret_frames:
                logger.warning(f"Failed after {not_ret} attempts")
                return None
            continue

        # Create numpy array from RGB image data
        numpy_image = raw_image.get_numpy_array()

        if numpy_image is None:
            logger.warning(f"no numpy image, attempt {not_ret+1}/{not_ret_frames+1}")
            time.sleep(wait_time)
            not_ret += 1
            if not_ret > not_ret_frames:
                logger.warning(f"Failed after {not_ret} attempts")
                return None
            continue

        logger.debug("Received a numpy image")
        return numpy_image

    return None


class DahengCamera:
    def __init__(self, device_idx: int = 0, lazy_calculations: bool = False,
                 return_data_tensor: bool = True):
        self.lazy_calculations = lazy_calculations
        self._camera_capture, self._device_manager = self.setup_camera(device_idx)
        self.acquisition_running = True
        gain_range_dict = self._camera_capture.Gain.get_range()
        self._gain_range = (gain_range_dict["min"], gain_range_dict["max"])

        exposure_range_dict = self._camera_capture.ExposureTime.get_range()
        self._exposure_range = (exposure_range_dict["min"], exposure_range_dict["max"])

        self.return_data_tensor = return_data_tensor
        if 'Mono8' in self.available_pixel_formats:
            self.color_data = False
        else:
            self.color_data = True

        self._available_exposure_modes = self._camera_capture.ExposureAuto.get_range()
        self._custom_autoexposure_active = False

    @staticmethod
    def setup_camera(device_idx: int = 0):
        device_manager: DeviceManager = DeviceManager()
        dev_num, dev_info_list = device_manager.update_device_list()
        # logger.debug(dev_info_list)
        if dev_num == 0:
            logger.debug("Haven't found any devices")
            raise ValueError

        # Open device
        # Get the list of basic device information
        serial_number: str = dev_info_list[device_idx].get("sn")
        # Open the device by serial number
        camera_capture = device_manager.open_device_by_sn(serial_number)
        logger.debug("DeviceManager initialized")

        # set continuous acquisition
        camera_capture.TriggerMode.set(GxSwitchEntry.OFF)
        # set the acq buffer count
        camera_capture.data_stream[0].set_acquisition_buffer_number(1)

        # Start acquisition
        camera_capture.stream_on()

        return camera_capture, device_manager


    @property
    def packed_pixel_format(self):
        return False

    @property
    def autoexposure_off_name(self):
        return "Off"

    @property
    def bit_depth(self):
        return PIXEL_FORMAT_BIT_DEPTH[self.available_pixel_formats[self.current_pixel_format]]

    @property
    def max_value(self):
        return (2 ** self.bit_depth) - 1

    @property
    def available_exposure_modes(self):
        return self._available_exposure_modes

    @property
    def gain_range(self):
        return self._gain_range

    @property
    def exposure_range(self):
        return self._exposure_range

    @property
    def max_exposure(self):
        return self.exposure_range[1]

    @property
    def min_exposure(self):
        return self.exposure_range[0]

    @property
    def max_gain(self):
        return self.gain_range[1]

    @property
    def min_gain(self):
        return self.gain_range[0]

    @property
    def max_width(self):
        return self._camera_capture.WidthMax.get()

    @property
    def min_width(self):
        return self._camera_capture.Width.get_range()['min']

    @property
    def inc_width(self):
        return self._camera_capture.Width.get_range()['inc']
    
    @property
    def inc_height(self):
        return self._camera_capture.Height.get_range()['inc']

    @property
    def min_height(self):
        return self._camera_capture.Height.get_range()['min']

    @property
    def max_height(self):
        return self._camera_capture.HeightMax.get()

    @property
    def available_pixel_formats(self):
        return self._camera_capture.PixelFormat.get_range()

    @property
    def current_exposure(self):
        return self._camera_capture.ExposureTime.get()

    @property
    def current_exposure_mode(self):
        return self._camera_capture.ExposureAuto.get()[1]

    @property
    def current_pixel_format(self):
        return self._camera_capture.PixelFormat.get()[1]

    @property
    def current_gain(self):
        return self._camera_capture.Gain.get()
    
    @property
    def width(self):
        return self._camera_capture.Width.get()
    
    @property
    def height(self):
        return self._camera_capture.Height.get()

    def start_acquisition(self):
        if not self.acquisition_running:
            self._camera_capture.stream_on()
            self.acquisition_running = True
    
    def stop_acquisition(self):
        if self.acquisition_running:
            self._camera_capture.stream_off()
            self.acquisition_running = False

    def set_custom_autoexposure(self, autoexposure_cls):
        if autoexposure_cls is None:
            self.turn_off_custom_autoexposure()
            return
        if self._custom_autoexposure_active:
            logger.info("Replacing custom autoexposure method")
        self._custom_autoexposure = autoexposure_cls(max_possible_value=self.max_value,
                                                     max_exposure_value=self.max_exposure)
        self._custom_autoexposure_active = True
        self.turn_off_camera_autoexposure()

    def turn_off_custom_autoexposure(self):
        if self._custom_autoexposure_active:
            self._custom_autoexposure = None
            self._custom_autoexposure_active = False

    def turn_off_camera_autoexposure(self):
        self.set_exposure_mode(self.autoexposure_off_name)

    def set_exposure_mode(self, exposure_mode: str):
        if self._custom_autoexposure_active and exposure_mode != self.autoexposure_off_name:
            logger.warning("Replacing the custom camera autoexposure with the one from camera")
            self.turn_off_custom_autoexposure()

        if exposure_mode not in self.available_exposure_modes:
            logger.warning(f"Exposure mode {exposure_mode} not in {self.available_exposure_modes}")
            return
        self._camera_capture.ExposureAuto.set(self.available_exposure_modes[exposure_mode])

    def set_exposure(self, exposure):
        if exposure < self.exposure_range[0] or exposure > self.exposure_range[1]:
            logger.warning(f"Exposure {exposure} not in range {self.exposure_range}")
            return
        self._camera_capture.ExposureTime.set(float(exposure))

    def set_gain(self, gain):
        if gain < self.gain_range[0] or gain > self.gain_range[1]:
            logger.warning(f"Gain {gain} not in range {self.gain_range}")
            return
        if not self._camera_capture.Gain.is_writable():
            logger.warning("Gain is not writable")
            return
        self._camera_capture.Gain.set(float(gain))

    @restart_acquisition_if_running
    def set_width(self, width: int):
        if not self._camera_capture.Width.is_writable():
            logger.warning("Width is not writable")
            return

        target = _nearest_valid_size_and_inc(width, min_v=self.min_width,
                                                  max_v=self.max_width, inc=self.inc_width)
        if target != width:
            logger.info(f"Adjusted width from {width} to {target} (min={self.min_width}, max={self.max_width}, inc={self.inc_width}).")

        self._camera_capture.Width.set(int(target))

    @restart_acquisition_if_running
    def set_height(self, height: int):
        if not self._camera_capture.Height.is_writable():
            logger.warning("Height is not writable")
            return

        target = _nearest_valid_size_and_inc(height, min_v=self.min_height,
                                                  max_v=self.max_height, inc=self.inc_height)
        if target != height:
            logger.info(f"Adjusted height from {height} to {target} (min={self.min_height}, max={self.max_height}, inc={self.inc_height}.")

        self._camera_capture.Height.set(int(target))

    @restart_acquisition_if_running
    def set_pixel_format(self, pixel_format: str):
        if pixel_format not in self.available_pixel_formats: # type: ignore
            raise ValueError(f"Incorrect depth value: {pixel_format}")
        
        pixel_format = self.available_pixel_formats[pixel_format]
        is_implemented = self._camera_capture.PixelFormat.is_implemented()
        if is_implemented:
            logger.debug("Camera PixelFormat is implemented")
            # Is this writable
            self._camera_capture.stream_off()
            is_writable = self._camera_capture.PixelFormat.is_writable()
            if is_writable:
                logger.debug("Camera PixelFormat is writable")
                # Set pixel format
                self._camera_capture.PixelFormat.set(pixel_format)
            self._camera_capture.stream_on()
            # Is this readable
            is_readable = self._camera_capture.PixelFormat.is_readable()
            if is_readable:
                # logger.warning pixel format
                logger.debug(f"Camera PixelFormat is readable. Currently - {self._camera_capture.PixelFormat.get()[1]}")

    def __next__(self):
        frame = get_numpy_image_from_stream(self._camera_capture.data_stream[0],
                                            self.current_exposure)
        if frame is None:
            return None

        if self._custom_autoexposure_active:
            new_exposure = self._custom_autoexposure(frame, self.current_exposure)
            self.set_exposure(new_exposure)

        if self.return_data_tensor:
            data_tensor = DataPixelTensor(frame, lazy_calculations=self.lazy_calculations,
                                          color_data=self.color_data, bit_depth=self.bit_depth)
            return data_tensor
        else:
            return frame


if __name__ == "__main__":
    import cv2
    live_camera = DahengCamera(lazy_calculations=True)

    for i in range(4):
        frame = next(live_camera)

        cv2.imwrite(f"frame_{i}.png", frame.raw)
