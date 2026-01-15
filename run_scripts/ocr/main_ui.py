import datetime
import logging
import argparse
import os
import sys
import types

# import PyQt5.QtCore
import PyQt5.QtGui
import PyQt5.QtWidgets
from PyQt5.QtWidgets import QMainWindow, QGridLayout, QWidget, QApplication, QStatusBar, QMessageBox
from PyQt5.QtCore import QTimer, QTime

import numpy as np
import cv2

from src.gui_config import Config

from src.gui import back_ui, dialog_save_photo, mywidgets, range_scale_window


########################################################################################################################
class MyWindow(QMainWindow):
    """ GUI main window """
    def __init__(self, args) -> None:
        super().__init__()

        logging.basicConfig(
            filename=f"{os.path.join(Config.log_path, datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S'))}___.log",
            level=Config.log_level,
            format='%(asctime)s | %(levelname)s | %(relativeCreated)d | %(thread)d | %(process)d | %(message)s'
        )
        logging.info(f"Configurations: {Config.__dict__}")

        # For sysem restart
        self.restart_time = QTime(14, 30)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_midnight)
        self.timer.start(1000)  
        
        # Open range scaling popup on Crtl+J
        self.range_scaling_action = PyQt5.QtWidgets.QAction(self)
        self.range_scaling_action.setShortcut('Ctrl+J')
        self.range_scaling_action.triggered.connect(self.open_range_scale_dialog)
        self.addAction(self.range_scaling_action)

        # Create the back end and set callbacks
        self.back_end = back_ui.BackUI(args.video, args.pos, args.obstacle_cache)
        self.back_end.status_cb = self.status_callback
        self.back_end.statusbar_cb = lambda s: self.status_bar.showMessage(s)
        self.back_end.image_cb = self.image_callback
        self.back_end.msgbox_cb = self.msgbox_callback

        available_geometry = PyQt5.QtGui.QGuiApplication.primaryScreen().availableGeometry()
        available_height: int = available_geometry.height()
        available_width: int = available_geometry.width()

        centre = available_geometry.center()
        with open('src/qss/dark.qss', 'r') as f:
            self.setStyleSheet(f.read())

        self.resize(available_width // 2, available_height // 2)
        self.frameGeometry().moveCenter(centre)
        self.setWindowTitle('People Track')

        self.msg_box = None   # Message box for "Preprocessing in Progress"
        self.range_scaling_window = None

        self.photo_name = ''

        # GUI elements
        self.status_bar: QStatusBar = QStatusBar()

        self.processing_controls = mywidgets.ProcessingControls(self)
        self.pane_camera = mywidgets.ImagePane('Camera', self)
        self.pane_processing = mywidgets.ImagePane('Processing', self)

        self.widget_exposure = mywidgets.FancySlider('Camera Exposure', Config.cam_min_exposure,
                                                     Config.cam_max_exposure, Config.cam_default_exposure, 0, self)

        self.widget_gain = mywidgets.FancySlider('Camera Gain', Config.cam_min_gain,
                                                 Config.cam_max_gain, Config.cam_default_gain, 0, self)

        self.widget_nz_scale = mywidgets.FancySlider('NZ scale', 0.5, 5.0, 1.8, 2, self)

        self.outer_layout: QGridLayout = QGridLayout()
        self.main_widget: QWidget = QWidget()

        self.init_window()
        self.set_backend_defaults()

    def set_backend_defaults(self):
        """We must have correct defaults, before any UI signal arrive"""
        self.back_end.set_synth_coord_x(0)
        self.back_end.set_synth_coord_y(0)
        self.back_end.current_synth_idx = 0

        # for i in range(4):
        #     self.back_end.set_roi(0, i)
        
        self.back_end.set_auto_exposure(self.processing_controls.get_auto_exposure())
        
        self.back_end.set_camera_exposure(self.widget_exposure.value())
        self.back_end.set_camera_gain(self.widget_gain.value())
        self.back_end.set_nz_scale(self.widget_nz_scale.value())

        self.back_end.set_image_crop(self.processing_controls.get_crop())
        self.back_end.set_obstacles_off(self.processing_controls.get_obstacles_off())

        self.back_end.set_range_scale(1.0)

    def init_window(self) -> None:
        # Status bar
        self.status_bar.setStyleSheet("background : lightblue;")
        self.setStatusBar(self.status_bar)  # Adding status bar to the main window
        self.status_bar.showMessage('Ready to start')

        self.processing_controls.synth_x_changed_signal.connect(lambda x: self.back_end.set_synth_coord_x(x))
        self.processing_controls.synth_y_changed_signal.connect(lambda y: self.back_end.set_synth_coord_y(y))
        self.processing_controls.change_synth_signal.connect(lambda: self.back_end.change_synth_image())
        self.processing_controls.take_photo_signal.connect(self.click_take_photo)
        self.processing_controls.crop_signal.connect(lambda b: self.back_end.set_image_crop(b))
        self.processing_controls.obstacles_off_signal.connect(lambda b: self.back_end.set_obstacles_off(b))
        self.processing_controls.auto_exposure_signal.connect(lambda s: self.back_end.set_auto_exposure(s))

        # Distance references
        self.processing_controls.y1_max_changed_signal.connect(lambda y: self.back_end.set_distance_refs(y,0))
        self.processing_controls.y2_min_changed_signal.connect(lambda y: self.back_end.set_distance_refs(y,1))
        
        # Roi
        self.processing_controls.roi_x1_changed_signal.connect(lambda n: self.back_end.set_roi(n, 0))
        self.processing_controls.roi_y1_changed_signal.connect(lambda n: self.back_end.set_roi(n, 1))
        self.processing_controls.roi_x2_changed_signal.connect(lambda n: self.back_end.set_roi(n, 2))
        self.processing_controls.roi_y2_changed_signal.connect(lambda n: self.back_end.set_roi(n, 3))
        
        # Save PATH
        self.processing_controls.directory_selected_signal.connect(lambda p: self.back_end.set_save_path(p))
        
        self.pane_camera.start_signal.connect(lambda is_recording: self.back_end.start_camera(is_recording))
        self.pane_camera.stop_signal.connect(lambda: self.back_end.stop_camera())
        self.pane_processing.start_signal.connect(lambda is_recording: self.back_end.start_processing(is_recording))
        self.pane_processing.stop_signal.connect(lambda: self.back_end.stop_processing())

        # Options: Exposure, gain etc.
        self.widget_exposure.update_signal.connect(lambda value: self.back_end.set_camera_exposure(value))
        self.widget_gain.update_signal.connect(lambda value: self.back_end.set_camera_gain(value))
        self.widget_nz_scale.update_signal.connect(lambda value: self.back_end.set_nz_scale(value))

        # grid y, x, h, w
        self.outer_layout.addWidget(self.pane_camera, 0, 0, 1, 2)
        self.outer_layout.addWidget(self.pane_processing, 0, 2, 1, 2)
        self.outer_layout.addWidget(self.processing_controls, 1, 1, 2, 2)
        self.outer_layout.addWidget(self.widget_exposure, 1, 0, 1, 1)
        self.outer_layout.addWidget(self.widget_gain, 2, 0, 1, 1)
        self.outer_layout.addWidget(self.widget_nz_scale, 1, 3, 1, 1)

        self.outer_layout.setColumnStretch(0, 5)
        self.outer_layout.setColumnStretch(1, 1)
        self.outer_layout.setColumnStretch(2, 1)
        self.outer_layout.setColumnStretch(3, 5)

        self.main_widget.setLayout(self.outer_layout)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.showMaximized()
        self.status_callback('READY')

    def closeEvent(self, event):
        """Close all windows and quit if this one is closed"""
        QApplication.instance().quit()

    ######### Calbacks to enable/disable GUI elements when the status changes
    def status_callback_btn_cam_proc(self, status: str):
        if status == 'READY':
            self.pane_camera.smart_enable('off')
            self.pane_processing.smart_enable('disabled')
        elif status == 'CAMERA':
            self.pane_camera.smart_enable('on')
            self.pane_processing.smart_enable('off')
        elif status == 'PROCESS':
            self.pane_camera.smart_enable('disabled')
            self.pane_processing.smart_enable('on')
        else:
            raise ValueError(f'Wrong status {status} !!!')

    def status_callback(self, status: str):
        """Called when the main status changes, all enable/disable logic must go through here !"""
        logging.debug(f'status_callback: status={status}')
        self.status_callback_btn_cam_proc(status)
        self.processing_controls.smart_enable(status)

    #### Other callbacks
    def image_callback(self, image: np.ndarray, label_name: str):
        """Update an image label with an OpenCV image"""
        if label_name == 'CAM':
            self.pane_camera.set_frame(image)
        else:
            self.pane_processing.set_frame(image)

    # Activates when Take photo button is clicked
    def click_take_photo(self):
        logging.debug(f"Taking photo")
        dlg = dialog_save_photo.SavePhotoDialog(self.photo_name)
        dlg.exec()
        self.photo_name = dlg.user_input_save.text()
        self.back_end.take_photo(self.photo_name)
        self.status_bar.showMessage('Photo is ready')

    # When ctrl+J is pressed ...
    def open_range_scale_dialog(self):
        self.range_scaling_window = range_scale_window.RangeScaleWindow(self.back_end)
        self.range_scaling_window.show()

    # slots
    # @pyqtSlot(str, str, str)
    # def show_message(self, text: str, title: str, message_type: str) -> None:
    #     dlg: types.FunctionType = message_types.get(message_type)
    #     dlg(self, title, text)

    def msgbox_callback(self, action: str):
        # print('msgbox_callback: action=', action)
        if action == 'SHOW':
            self.msg_box = QMessageBox()
            self.msg_box.setWindowTitle('Information')
            self.msg_box.setText('Preprocessing in Progress. Please Wait')
            self.msg_box.exec()
        elif action == 'CLOSE' and self.msg_box is not None:
            self.msg_box.accept()

    # Check time for system restart
    def check_midnight(self):
        current_time = QTime.currentTime()
        if current_time.hour() == 0 and current_time.minute() == 0 and current_time.second() == 0:

            self.back_end.stop_processing()
            self.back_end.start_processing(False)

########################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='People Track GUI code')
    parser.add_argument('-v', '--video', type=str, help='Run a video file or a folder with PXIs, no camera')
    parser.add_argument('-p', '--pos', type=int, help='Starting position')
    parser.add_argument('-c', '--obstacle-cache', action='store_true',
                        help='Cache static obstacles to avoid running DINO+SAM every time')
    args = parser.parse_args()
    print('args=', args)

    app: QApplication = QApplication(sys.argv)
    # Set theme


    win: MyWindow = MyWindow(args)
    win.show()
    sys.exit(app.exec())
