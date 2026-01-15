# By Oleksiy Grechnyev
# Backend for the MyWindow class
# It uses Qt for threads, signals + slots, but no gui elements
# So it depends on Qt (not a 100% pure back end), but not too much
# All worker threads (camera processing, etc.) are managed here

import os
import sys

import numpy as np

import logging

import PyQt6.QtCore
from PyQt6.QtCore import QThread, Qt, pyqtSlot, pyqtSignal

from src.threads.thread_camera import Camera264Thread
from src.threads.thread_processing import ProcessingWorker

########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
class BackUI(PyQt6.QtCore.QObject):

    points_sig = pyqtSignal(list)
    stop_reconstruction_signal = pyqtSignal()
    reconstruction_ready = pyqtSignal(object)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.current_status = None

        self.raw_video_dir = os.path.join("videos", "raw")
        self.processed_video_dir = os.path.join("videos", "processed")

        self.save_dir = "D:/work/dx_vyzai_people_track/save"
        
        self.camera_thread: Camera264Thread | None = None
        
        self.proc_thread = QThread(self)
        self.processing_worker = ProcessingWorker()
        self.processing_worker.moveToThread(self.proc_thread)
        
        self.proc_thread.started.connect(self.processing_worker.initialize)

        self.points_sig.connect(
            self.processing_worker.init_sam_track,
            Qt.ConnectionType.QueuedConnection
        )

        self.stop_reconstruction_signal.connect(
            self.processing_worker.stop_reconstruction,
            Qt.ConnectionType.DirectConnection
        )

        self.processing_worker.vis_frame_ready.connect(
            self.on_cam_frame, Qt.ConnectionType.QueuedConnection
        )

        self.processing_worker.reconstruction_ready.connect(
            self.on_reconstruction, Qt.ConnectionType.QueuedConnection
        )

        self.sam2_points_data = None
        
        self.status_cb = None    
        self.statusbar_cb = None   
        self.image_cb = None   
        self.scene_cb = None
        
        self.proc_thread.start()

    def status_changed(self, status:str):
        self.current_status = status
        if self.status_cb is not None:
            self.status_cb(status)
    
    def statusbar_changed(self, status:str):
        if self.statusbar_cb is not None:
            self.statusbar_cb(status)

    #### Simple Setters
    
        
    ##### Other logic
    # def take_photo(self, name: str):
    #     self.camera_thread.take_photo(name) if self.camera_thread else logging.debug(
    #         f"Camera is not running")
    #     self.processing_worker.take_photo(name) if self.processing_worker else logging.debug(
    #         f"Processing not started")

    def start_camera(self):
        
        assert self.camera_thread is None
        logging.debug(f"Starting camera video")
        self.statusbar_changed('Starting Camera...')


        self.camera_thread: Camera264Thread = Camera264Thread(self.video_path)

        self.camera_thread.change_pixmap_signal.connect(lambda img: self.image_cb(img, 'CAM'))
        self.camera_thread.change_pixmap_signal.connect(
            self.processing_worker.on_frame, Qt.ConnectionType.QueuedConnection
        )
        
        logging.debug(f"Created camera instance")

        if None:
            self.raw_video_writer_thread = VideoWriterThread(self.raw_video_dir, crop_fov=self.image_crop)
            self.camera_thread.change_pixmap_signal.connect(
                lambda x: self.raw_video_writer_thread.update_frame_queue(x[0])
            )
            self.raw_video_writer_thread.start()
        else:
            self.raw_video_writer_thread = None

        # start the thread
        self.camera_thread.start()
        
        
        logging.info(f"Camera instance started")
        self.status_changed('CAMERA')

    def stop_camera(self):
        if self.camera_thread is None:
            return

        logging.debug(f"Stopping camera video")
        # stop the thread and disconnect signals
        self.camera_thread.stop()
        self.camera_thread.change_pixmap_signal.disconnect()
        logging.info(f"Camera thread stopped")
        self.camera_thread = None

        # if self.raw_video_writer_thread is not None:
        #     self.raw_video_writer_thread.stop()
        #     self.raw_video_writer_thread = None

        self.statusbar_changed('Ready to start')
        self.status_changed('READY')
    
    @pyqtSlot(list)
    def set_points_data(self, points: list):
        self.points_sig.emit(points)
    
    @pyqtSlot()
    def stop_reconstruction(self):
        self.stop_reconstruction_signal.emit()
        print('back_ui')


    @pyqtSlot(object)
    def on_reconstruction(self, data):
        print('ready')

        self.reconstruction_ready.emit(data)


    @pyqtSlot(object)
    def on_cam_frame(self, img):
        # кадр від камери у ліве вікно
        if self.image_cb:
            self.image_cb(img, 'CAM')
            
    # def start_processing(self, is_recording: bool):
    #     assert self.processing_worker is None

    #     logging.debug(f"Starting processing")
    #     self.statusbar_changed('Starting Processing...')

    #     # Create instance and connect signals
    #     self.processing_worker: ProcessingThread = ProcessingThread(crop_fov=self.image_crop,
    #                                                                 obstacles_off=self.obstacles_off,
    #                                                                 use_obstacle_cache=self.use_obstacle_cache,
    #                                                                 video_path=self.video_path,
    #                                                                 roi=self.roi,
    #                                                                 save_dir=self.save_dir,
    #                                                                 dist_refs = self.dist_refs
    #                                                                 )
        
    #     self.processing_worker.batch_yield_signal.connect(lambda img: self.image_cb(img, 'PROC'))
    #     self.processing_worker.close_message_box_signal.connect(lambda : self.msgbox_cb('CLOSE'))
    #     self.camera_thread.change_pixmap_signal.connect(self.processing_worker.update_image_for_processing)
    #     self.update_range_scale_signal.connect(self.processing_worker.set_range_scale)

    #     if is_recording:
    #         self.processed_video_writer_thread = VideoWriterThread(self.processed_video_dir,
    #                                                                 crop_fov=self.image_crop)
    #         self.processing_worker.batch_yield_signal.connect(
    #             lambda x: self.processed_video_writer_thread.update_frame_queue(x)
    #         )
    #         self.processed_video_writer_thread.start()
    #     else:
    #         self.processed_video_writer_thread = None

    #     logging.debug(f"Created processing instance")

    #     # start the thread
    #     self.processing_worker.start()
    #     logging.info(f"Processing instance started")

    #     # Pass parameters to teh newly created processing thread
    #     self.update_range_scale_signal.emit(self.range_scale)

    #     self.msgbox_cb('SHOW')
    #     self.status_changed('PROCESS')

    # def stop_processing(self):
    #     if self.processing_worker is None:
    #         return
    #     logging.debug(f"Stopping processing")

    #     self.camera_thread.change_pixmap_signal.disconnect(self.processing_worker.update_image_for_processing)

    #     # stop the thread and disconnect signals
    #     self.processing_worker.stop()
    #     self.processing_worker.batch_yield_signal.disconnect()
    #     self.processing_worker = None

    #     logging.info(f"Processing thread stopped")

    #     if self.processed_video_writer_thread is not None:
    #         self.processed_video_writer_thread.stop()

    #     self.statusbar_changed('Processing stopped')
    #     self.status_changed('CAMERA')
    

########################################################################################################################
