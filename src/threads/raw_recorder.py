import queue
import threading
from typing import Literal, Union
from pathlib import Path

import numpy as np
from loguru import logger

from src.data_pixel_tensor import DataPixelTensor


class DataRecorder:
    def __init__(self, save_path: Path, frame_queue: queue.Queue,
                 stop_event: threading.Event):
        self.save_path = save_path
        self.frame_queue = frame_queue
        self.stop_event = stop_event

        self.save_path.mkdir(parents=True, exist_ok=True)
        self.current_save_idx = 0

    def _save_numpy(self, data: Union[DataPixelTensor, np.ndarray]):
        if isinstance(data, DataPixelTensor):
            data = data.raw

        filename = self.save_path / f"{self.save_path.stem}-{self.current_save_idx:03d}.npy"
        np.save(filename, data)

    def _save_pxi(self, data_tensor: DataPixelTensor):
        filename = self.save_path / f"{self.save_path.stem}-{self.current_save_idx:03d}.pxi"
        data_tensor.save_pxi(filename)

    def _save_tif(self, data_tensor: DataPixelTensor):
        filename = self.save_path / f"{self.save_path.stem}-{self.current_save_idx:03d}.tif"
        data_tensor.save_tif(filename)

    def start_recording_loop(self, save_format: Literal[".pxi", ".tif", ".npy"] = ".npy"):
        while not self.stop_event.is_set():
            try:
                data_tensor = self.frame_queue.get(block=True, timeout=.1)  # Wait for items 

                if data_tensor is None:  # Check for sentinel value
                    logger.warning("Saver received stop signal.")
                    break

                if save_format == '.npy':
                    self._save_numpy(data_tensor)
                elif save_format == ".tif":
                    self._save_tif(data_tensor)
                elif save_format == '.pxi':
                    self._save_pxi(data_tensor)
                else:
                    raise ValueError(f"Incorrect save format: {save_format}")

                logger.debug(f"Saved frame {self.current_save_idx} in the format {save_format}")

                self.current_save_idx += 1

                # frame_queue.task_done() # Signal that the item processing is complete
            except queue.Empty:
                # Queue was empty during timeout, check if we should stop
                continue # Continue waiting for frames

            except Exception as e:
                logger.error(f"Error in saver: {e}")
                # Potentially break or log error, depending on severity
                break


def record_pxi_loop(save_path: Path, frame_queue: queue.Queue):
    current_save_idx = 0
    while True:
        try:
            data_tensor = frame_queue.get(block=True, timeout=.1) # Wait for items

            if data_tensor is None:  # Check for sentinel value
                logger.warning("Saver received stop signal.")
                break

            filename = save_path / f"{save_path.stem}-{current_save_idx:03d}.pxi"

            data_tensor.save_pxi(filename)
            current_save_idx += 1

            # frame_queue.task_done() # Signal that the item processing is complete
        except queue.Empty:
            # Queue was empty during timeout, check if we should stop
            continue # Continue waiting for frames

        except Exception as e:
            logger.error(f"Error in saver: {e}")
            # Potentially break or log error, depending on severity
            break


def record_numpy_loop(save_path: Path, frame_queue: queue.Queue,
                      stop_event: threading.Event):

    current_save_idx = 0
    while not stop_event.is_set():
        try:
            data_tensor = frame_queue.get(block=True, timeout=.1)  # Wait for items

            if data_tensor is None:  # Check for sentinel value
                logger.warning("Saver received stop signal.")
                break

            filename = save_path / f"{save_path.stem}-{current_save_idx:03d}.npy"
            np.save(filename, data_tensor)
            logger.debug("Saved .npy file")

            current_save_idx += 1

            # frame_queue.task_done() # Signal that the item processing is complete
        except queue.Empty:
            # Queue was empty during timeout, check if we should stop
            continue # Continue waiting for frames

        except Exception as e:
            logger.error(f"Error in saver: {e}")
            # Potentially break or log error, depending on severity
            break