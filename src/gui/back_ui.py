# By Oleksiy Grechnyev
# Backend for the MyWindow class
# It uses Qt for threads, signals + slots, but no gui elements
# So it depends on Qt (not a 100% pure back end), but not too much
# All worker threads (camera processing, etc.) are managed here

import os
import sys

import numpy as np

import logging

import PyQt5.QtCore


from src.gui_config import Config
from src.threads.legacy.thread_processing import ProcessingThread
from src.threads.legacy.thread_video_writer import VideoWriterThread


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
class BackUI(PyQt5.QtCore.QObject):
    update_exposure_signal = PyQt5.QtCore.pyqtSignal(float)
    update_auto_exposure_signal = PyQt5.QtCore.pyqtSignal(str)
    update_gain_signal = PyQt5.QtCore.pyqtSignal(float)
    update_nz_scale_signal = PyQt5.QtCore.pyqtSignal(float)
    update_overlay_signal = PyQt5.QtCore.pyqtSignal(str)
    update_overlay_coords_signal = PyQt5.QtCore.pyqtSignal(tuple)
    update_range_scale_signal = PyQt5.QtCore.pyqtSignal(float)
    
    update_roi_coords_signal = PyQt5.QtCore.pyqtSignal(list)

    def __init__(self, video_path: str, start_pos: int, use_obstacle_cache: bool):
        super().__init__()
        self.video_path = video_path
        self.start_pos = start_pos
        self.use_obstacle_cache = use_obstacle_cache
        self.current_status = None
        
        # Synth image overlay and camera parameters
        self.synth_coords = [0, 0]
        self.current_synth_idx = 0
        self.exposure_value = 0
        self.gain_value = 0
        self.nz_scale_value = 0
        self.range_scale = 1.0
        
        self.raw_video_dir = os.path.join("videos", "raw")
        self.processed_video_dir = os.path.join("videos", "processed")

        self.save_dir = "D:/work/dx_vyzai_people_track/save"
        
        self.camera_thread = None
        self.processing_thread = None
        self.raw_video_writer_thread = None
        self.processed_video_writer_thread = None
        self.image_crop: bool = False
        self.obstacles_off: bool = False
        self.auto_exposure: str = 'OFF'
        
        self.status_cb = None      # Status callback
        self.statusbar_cb = None   # Status bar callback
        self.image_cb = None       # Image callback
        self.msgbox_cb = None       # Message box callback
        
        # Roi
        self.roi = [0,0,0,0]
        
        # Distance references
        self.dist_refs = [0,0]
        
    def status_changed(self, status:str):
        self.current_status = status
        if self.status_cb is not None:
            self.status_cb(status)
    
    def statusbar_changed(self, status:str):
        if self.statusbar_cb is not None:
            self.statusbar_cb(status)

    #### Simple Setters
    def set_image_crop(self, checked: bool):
        self.image_crop = checked

    def set_obstacles_off(self, checked: bool):
        self.obstacles_off = checked
        
    def set_auto_exposure(self, value: str):
        self.auto_exposure = value
        self.update_auto_exposure_signal.emit(self.auto_exposure)
        
    def set_camera_exposure(self, exposure_value: int):
        self.exposure_value = float(exposure_value)
        self.update_exposure_signal.emit(self.exposure_value)
        logging.debug(f"exposure emitted {exposure_value}")
        
    def set_camera_gain(self, gain_value: int):
        self.gain_value = float(gain_value)
        self.update_gain_signal.emit(self.gain_value)
        logging.debug(f"gain emitted {gain_value}")

    def set_nz_scale(self, value: float):
        self.nz_scale_value = value
        self.update_nz_scale_signal.emit(self.nz_scale_value)
        logging.debug(f"nz scale emitted {value}")

    def set_range_scale(self, value: float):
        self.range_scale = value
        self.update_range_scale_signal.emit(self.range_scale)
        
    def set_synth_coord_x(self, x):
        self.synth_coords[0] = x
        self.update_overlay_coords_signal.emit(tuple(self.synth_coords))
        logging.debug(f"coords emitted {self.synth_coords}")
    
    def set_synth_coord_y(self, y):
        self.synth_coords[1] = y
        self.update_overlay_coords_signal.emit(tuple(self.synth_coords))
        logging.debug(f"coords emitted {self.synth_coords}")
        
    def change_synth_image(self):
        self.current_synth_idx += 1
        if self.current_synth_idx > len(Config.overlay_images) - 1:
            self.current_synth_idx = 0

        img_name = Config.overlay_images[self.current_synth_idx]
        self.update_overlay_signal.emit(img_name)
        logging.debug(f"overlay_image emitted {img_name}")

    def set_roi(self, x, index):
        self.roi[index] = x
        self.update_roi_coords_signal.emit(self.roi)
        logging.debug(f"roi point:{index} to:{x} ({self.roi})")
        
    def set_save_path(self, p):
        self.save_dir = p
        logging.debug(f"Save path: {p}")
        
    def set_distance_refs(self, dist, index):
        self.dist_refs[index] = dist
        logging.debug(f"Distance reference:{index} to:{dist} ({self.roi})")
        
    ##### Other logic
    def take_photo(self, name: str):
        self.camera_thread.take_photo(name) if self.camera_thread else logging.debug(
            f"Camera is not running")
        self.processing_thread.take_photo(name) if self.processing_thread else logging.debug(
            f"Processing not started")

    def start_camera(self, is_recording: bool):
        assert self.camera_thread is None
        logging.debug(f"Starting camera video")
        self.statusbar_changed('Starting Camera...')

        # Create instance and connect signals
        if self.video_path is not None:
            # Work with video or a directory of PXI files (dummy thread)
            from src.threads.legacy.thread_camera_dummy import CameraDummy264Thread
            self.camera_thread: CameraDummy264Thread = CameraDummy264Thread(self.video_path, self.start_pos,
                                                                            crop_fov=self.image_crop)
        else:
            # Work with polarization camera (requires one obviously, and python drivers installed)
            from src.threads.legacy.thread_camera import Camera264Thread
            self.camera_thread: Camera264Thread = Camera264Thread(crop_fov=self.image_crop)

        self.camera_thread.change_pixmap_signal.connect(lambda img: self.image_cb(img.view_img, 'CAM'))
        self.update_exposure_signal.connect(self.camera_thread.set_exposure)
        self.update_auto_exposure_signal.connect(self.camera_thread.set_auto_exposure)
        self.update_gain_signal.connect(self.camera_thread.set_gain)
        self.update_nz_scale_signal.connect(self.camera_thread.set_nz_scale)
        self.update_overlay_signal.connect(self.camera_thread.set_overlay_img)
        self.update_overlay_coords_signal.connect(self.camera_thread.set_overlay_coords)
        
        self.update_roi_coords_signal.connect(self.camera_thread.set_roi)
        
        logging.debug(f"Created camera instance")

        if is_recording:
            self.raw_video_writer_thread = VideoWriterThread(self.raw_video_dir, crop_fov=self.image_crop)
            self.camera_thread.change_pixmap_signal.connect(
                lambda x: self.raw_video_writer_thread.update_frame_queue(x.view_img)
            )
            self.raw_video_writer_thread.start()
        else:
            self.raw_video_writer_thread = None

        # start the thread
        self.camera_thread.start()
        
        # Pass all camera+synth obstacle parameters to the newly created camera thread
        self.update_exposure_signal.emit(self.exposure_value)
        self.update_auto_exposure_signal.emit(self.auto_exposure)
        self.update_gain_signal.emit(self.gain_value)
        self.update_nz_scale_signal.emit(self.nz_scale_value)
        self.update_overlay_coords_signal.emit(tuple(self.synth_coords))
        img_name = Config.overlay_images[self.current_synth_idx]
        self.update_overlay_signal.emit(img_name)
        
        self.update_roi_coords_signal.emit(self.roi)
        
        logging.info(f"Camera instance started")
        self.status_changed('CAMERA')

    def stop_camera(self):
        if self.camera_thread is None:
            return

        logging.debug(f"Stopping camera video")
        # stop the thread and disconnect signals
        self.update_exposure_signal.disconnect(self.camera_thread.set_exposure)
        self.update_auto_exposure_signal.disconnect(self.camera_thread.set_auto_exposure)
        self.update_gain_signal.disconnect(self.camera_thread.set_gain)
        self.update_nz_scale_signal.disconnect(self.camera_thread.set_nz_scale)
        self.update_overlay_signal.disconnect(self.camera_thread.set_overlay_img)
        self.update_overlay_coords_signal.disconnect(self.camera_thread.set_overlay_coords)
        self.camera_thread.stop()
        self.camera_thread.change_pixmap_signal.disconnect()
        logging.info(f"Camera thread stopped")
        self.camera_thread = None

        if self.raw_video_writer_thread is not None:
            self.raw_video_writer_thread.stop()
            self.raw_video_writer_thread = None

        self.statusbar_changed('Ready to start')
        self.status_changed('READY')

    def start_processing(self, is_recording: bool):
        assert self.processing_thread is None

        logging.debug(f"Starting processing")
        self.statusbar_changed('Starting Processing...')

        # Create instance and connect signals
        self.processing_thread: ProcessingThread = ProcessingThread(crop_fov=self.image_crop,
                                                                    obstacles_off=self.obstacles_off,
                                                                    use_obstacle_cache=self.use_obstacle_cache,
                                                                    video_path=self.video_path,
                                                                    roi=self.roi,
                                                                    save_dir=self.save_dir,
                                                                    dist_refs = self.dist_refs
                                                                    )
        
        self.processing_thread.batch_yield_signal.connect(lambda img: self.image_cb(img, 'PROC'))
        self.processing_thread.close_message_box_signal.connect(lambda : self.msgbox_cb('CLOSE'))
        self.camera_thread.change_pixmap_signal.connect(self.processing_thread.update_image_for_processing)
        self.update_range_scale_signal.connect(self.processing_thread.set_range_scale)

        if is_recording:
            self.processed_video_writer_thread = VideoWriterThread(self.processed_video_dir,
                                                                    crop_fov=self.image_crop)
            self.processing_thread.batch_yield_signal.connect(
                lambda x: self.processed_video_writer_thread.update_frame_queue(x)
            )
            self.processed_video_writer_thread.start()
        else:
            self.processed_video_writer_thread = None

        logging.debug(f"Created processing instance")

        # start the thread
        self.processing_thread.start()
        logging.info(f"Processing instance started")

        # Pass parameters to teh newly created processing thread
        self.update_range_scale_signal.emit(self.range_scale)

        self.msgbox_cb('SHOW')
        self.status_changed('PROCESS')

    def stop_processing(self):
        if self.processing_thread is None:
            return
        logging.debug(f"Stopping processing")

        self.camera_thread.change_pixmap_signal.disconnect(self.processing_thread.update_image_for_processing)

        # stop the thread and disconnect signals
        self.processing_thread.stop()
        self.processing_thread.batch_yield_signal.disconnect()
        self.processing_thread = None

        logging.info(f"Processing thread stopped")

        if self.processed_video_writer_thread is not None:
            self.processed_video_writer_thread.stop()

        self.statusbar_changed('Processing stopped')
        self.status_changed('CAMERA')
    

########################################################################################################################
