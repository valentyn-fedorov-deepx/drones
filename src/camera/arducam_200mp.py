import time
from enum import IntEnum

from loguru import logger
import cv2
import numpy as np
from tqdm import tqdm

from src.camera.autofocus.base_autofocus import AutofocusController
from src.camera.arducam_lib.ArducamUvcXU import Devices, i2cWriteReg, i2cReadReg


MIN_EXPOSURE = -11
MAX_EXPOSURE = -1
MIN_FOCUS = 0
MAX_FOCUS = 1023


class Camera:
    def __init__(self, index=0, selector=cv2.CAP_ANY) -> None:
        self.index = index
        self.selector = selector
        self.cap = None
        self.width = None
        self.height = None
        self.fps = None
        self._custom_autofocus_active = False
        self._custom_autofocus = None

    def open(self):
        self.cap = cv2.VideoCapture(self.index, self.selector)
        if self.width and self.height:
            self.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if self.fps:
            self.set(cv2.CAP_PROP_FPS, self.fps)

    @property
    def current_focus(self):
        return self.cap.get(cv2.CAP_PROP_FOCUS)

    def set_width(self, width):
        self.width = width
        if self.cap is not None:
            self.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)

    def set_height(self, height):
        self.height = height
        if self.cap is not None:
            self.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def set_fps(self, fps):
        self.fps = fps
        if self.cap is not None:
            self.set(cv2.CAP_PROP_FPS, self.fps)

    def set_focus(self, val):
        if self.cap is not None:
            self.set(cv2.CAP_PROP_FOCUS, val)

    def set(self, selector, val):
        if self.cap is not None:
            self.cap.set(selector, val)

    def read(self):
        return self.cap.read()

    def reStart(self):
        self.release()
        time.sleep(0.5)
        self.open()

    def set_custom_autofocus(self):
        self._custom_autofocus = AutofocusController(MIN_FOCUS, MAX_FOCUS)
        self._custom_autofocus_active = True

    def turn_off_custom_autofocus(self):
        self._custom_autofocus_active = False
        self._custom_autofocus = None

    def release(self):
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()

    def __next__(self):
        ret, frame = self.cap.read()

        if self._custom_autofocus_active:
            new_focus = self._custom_autofocus(frame, self.current_focus)

            self.set_focus(new_focus)

        return ret, frame


class i2c_mode(IntEnum):
    I2C_MODE_8_8 = 1
    I2C_MODE_8_16 = 2
    I2C_MODE_16_8 = 3
    I2C_MODE_16_16 = 4


class CameraXU:
    def __init__(self) -> None:
        self.fd = -1
        self.Device = Devices()
        self.i2c_addr = 0x20

    def open(self, device_name):
        for i in self.Device:
            if i["name"] == device_name:
                self.fd = self.Device.open(i["id"])

    def read_eeprom(self, inf_eeprom_data_path="inf_eeprom_data.dat",
                    mac_eeprom_data_path="mac_eeprom_data.dat",
                    eeprom_data_len=4608):
        eeprom_addr = 0xA2
        inf_eeprom_data_reg = 0x3956
        mac_eeprom_data_reg = 0x4B56
        inf_eeprom_data = []
        mac_eeprom_data = []
        start_time = time.time()
        for i in tqdm(range(eeprom_data_len), desc='reading eeprom data ...'):
            inf_eeprom_data.append(self.read_register(eeprom_addr,
                                                      inf_eeprom_data_reg + i,
                                                      i2c_mode.I2C_MODE_16_8))
            mac_eeprom_data.append(self.read_register(eeprom_addr,
                                                      mac_eeprom_data_reg + i,
                                                      i2c_mode.I2C_MODE_16_8))

        inf_eeprom_data = np.array(inf_eeprom_data, dtype=np.uint8)
        mac_eeprom_data = np.array(mac_eeprom_data, dtype=np.uint8)
        inf_eeprom_data.tofile(inf_eeprom_data_path)
        mac_eeprom_data.tofile(mac_eeprom_data_path)
        logger.debug("save eeprom data to file {0} and {1}".format(inf_eeprom_data_path,
                                                            mac_eeprom_data_path))
        end_time = time.time()
        logger.debug("read eeprom data time: {0:.2f}s".format(end_time - start_time))

    def refresh(self):
        self.Device.refresh()
        # return [f"camera {i}" for i, cam in enumerate(self.Device)]
        return ["{0}".format(i["name"]) for i in self.Device]

    def read_register(self, i2c_addr, addr, mode):
        ret, val = i2cReadReg(self.fd, mode, i2c_addr, addr)
        if ret != 0:
            logger.error("Error reading register: ", ret)
            return None
        return val

    def write_register(self, i2c_addr, addr, value, mode):
        i2cWriteReg(self.fd, mode, i2c_addr, addr, value)

    def close(self):
        self.Device.close()
        self.fd = -1
