# By Oleksiy Grechnyev

import os
import sys

import numpy as np
import cv2 as cv

import logging

import PyQt5.QtWidgets
import PyQt5.QtCore
import PyQt5.QtGui


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
class FancySlider(PyQt5.QtWidgets.QFrame):
    """Slider joined with text field, plus label and default button"""
    update_signal = PyQt5.QtCore.pyqtSignal(float)
    
    def __init__(self, name, sl_min, sl_max, sl_default, n_digits: int, parent):
        super().__init__(parent)
        self.sl_min = sl_min
        self.sl_max = sl_max
        self.sl_default = sl_default
        self.num_type = int if n_digits == 0 else float
        self.n_digits = n_digits
        self.factor = 10 ** n_digits
        
        # Widgets
        self.slider = PyQt5.QtWidgets.QSlider(PyQt5.QtCore.Qt.Orientation.Horizontal, self)
        self.slider.setRange(int(sl_min * self.factor), int(sl_max * self.factor))
        self.slider.setTickPosition(PyQt5.QtWidgets.QSlider.TickPosition.TicksAbove)
        self.slider.valueChanged.connect(lambda value: self.update_value(value if self.n_digits == 0 else value / self.factor), 1)
        
        self.label = PyQt5.QtWidgets.QLabel(f'{name}    ({sl_min}-{sl_max})', self)
        self.btn_default = PyQt5.QtWidgets.QPushButton('Default', self)
        self.btn_default.clicked.connect(lambda: self.update_value(self.sl_default, 0))
        
        self.text_edit = PyQt5.QtWidgets.QLineEdit(self)
        if n_digits == 0:
            validator = PyQt5.QtGui.QIntValidator(sl_min, sl_max)
        else:
            validator = PyQt5.QtGui.QDoubleValidator(sl_min, sl_max, n_digits)

        self.text_edit.setValidator(validator)
        self.text_edit.editingFinished.connect(lambda: self.update_value(self.text_edit.text(), 2))

        # Set default value
        self.update_value(sl_default)
        
        # Layout
        self.my_layout = PyQt5.QtWidgets.QGridLayout()

        if True:
            # 1-row layout
            self.my_layout.addWidget(self.label, 0, 0, 1, 1)
            self.my_layout.addWidget(self.slider, 0, 1, 1, 1)
            self.my_layout.addWidget(self.btn_default, 0, 3, 1, 1)
            self.my_layout.addWidget(self.text_edit, 0, 2, 1, 1)
            self.my_layout.setColumnStretch(0, 2)
            self.my_layout.setColumnStretch(1, 3)
            self.my_layout.setColumnStretch(2, 1)
            self.my_layout.setColumnStretch(3, 1)
        else:
            # 2-row layout
            self.my_layout.addWidget(self.label, 0, 0, 1, 1)
            self.my_layout.addWidget(self.slider, 1, 0, 1, 1)
            self.my_layout.addWidget(self.btn_default, 0, 1, 1, 1)
            self.my_layout.addWidget(self.text_edit, 1, 1, 1, 1)
            self.my_layout.setColumnStretch(0, 3)
            self.my_layout.setColumnStretch(1, 1)

        self.setLayout(self.my_layout)
        self.setFrameShape(PyQt5.QtWidgets.QFrame.Box)
        self.setLineWidth(3)

    def update_value(self, value, src=0):
        value = self.num_type(value)
        if src != 1:
            self.slider.blockSignals(True)
            self.slider.setValue(int(value * self.factor))
            self.slider.blockSignals(False)
        if src != 2:
            self.text_edit.blockSignals(True)
            self.text_edit.setText(str(value))
            self.text_edit.blockSignals(False)
        self.update_signal.emit(float(value))

    def value(self):
        v = self.num_type(self.text_edit.text())
        return v
        

########################################################################################################################
class ImagePane(PyQt5.QtWidgets.QFrame):
    """An image pane with 3 buttons: start, start (no recording), stop"""
    start_signal = PyQt5.QtCore.pyqtSignal(bool)
    stop_signal = PyQt5.QtCore.pyqtSignal()

    def __init__(self, name, parent):
        super().__init__(parent)
        self.name = name

        # GUI elements
        self.video_label = PyQt5.QtWidgets.QLabel(self)
        self.video_label.setStyleSheet("background : black;")
        self.btn_start1 = PyQt5.QtWidgets.QPushButton(f'Start {name}', self)
        self.btn_start1.clicked.connect(lambda: self.start_signal.emit(True))
        self.btn_start2 = PyQt5.QtWidgets.QPushButton(f'Start {name} without recording', self)
        self.btn_start2.clicked.connect(lambda: self.start_signal.emit(False))
        self.btn_stop = PyQt5.QtWidgets.QPushButton(f'Stop {name}', self)
        self.btn_stop.clicked.connect(lambda: self.stop_signal.emit())

        # Layout
        self.my_layout = PyQt5.QtWidgets.QGridLayout()
        self.my_layout.addWidget(self.video_label, 0, 0, 1, 3)
        self.my_layout.addWidget(self.btn_start1, 1, 0, 1, 1)
        self.my_layout.addWidget(self.btn_start2, 1, 1, 1, 1)
        self.my_layout.addWidget(self.btn_stop, 1, 2, 1, 1)
        self.my_layout.setRowStretch(0, 5)
        self.my_layout.setRowStretch(1, 1)

        self.setLayout(self.my_layout)
        self.setFrameShape(PyQt5.QtWidgets.QFrame.Box)
        self.setLineWidth(3)

    def smart_enable(self, mode: str):
        if mode == 'disabled':
            self.btn_start1.setEnabled(False)
            self.btn_start2.setEnabled(False)
            self.btn_stop.setEnabled(False)
        elif mode == 'off':
            self.btn_start1.setEnabled(True)
            self.btn_start2.setEnabled(True)
            self.btn_stop.setEnabled(False)
        elif mode == 'on':
            self.btn_start1.setEnabled(False)
            self.btn_start2.setEnabled(False)
            self.btn_stop.setEnabled(True)
        else:
            raise ValueError('Wrong mode=', mode)

    def convert_cv_qt(self, rgb_image: np.ndarray) -> PyQt5.QtGui.QPixmap:
        """Convert an opencv RGB (!) image to QPixmap"""
        height, width, channel = rgb_image.shape
        bytes_per_line: int = channel * width
        convert_to_qt_format = PyQt5.QtGui.QImage(
            rgb_image.data, width, height, bytes_per_line,  PyQt5.QtGui.QImage.Format.Format_RGB888
        )
        qt_scaled_image = convert_to_qt_format.scaled(
            self.video_label.width(), self.video_label.height(), PyQt5.QtCore.Qt.AspectRatioMode.KeepAspectRatio, PyQt5.QtCore.Qt.SmoothTransformation
        )
        return PyQt5.QtGui.QPixmap.fromImage(qt_scaled_image)

    def set_frame(self, frame: np.ndarray):
        """Update with an OpenCV BGR frame"""
        logging.debug(f"Updating image for {self.name}")
        rgb_image: np.ndarray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        qt_img: PyQt5.QtGui.QPixmap = self.convert_cv_qt(rgb_image)
        self.video_label.setPixmap(qt_img)
        logging.debug(f"Updated image for {self.name}")


########################################################################################################################
class ProcessingControls(PyQt5.QtWidgets.QFrame):
    """A widget with processing+synth obstacle controls"""
    synth_x_changed_signal = PyQt5.QtCore.pyqtSignal(int)
    synth_y_changed_signal = PyQt5.QtCore.pyqtSignal(int)
    
    y1_max_changed_signal = PyQt5.QtCore.pyqtSignal(int)
    y2_min_changed_signal = PyQt5.QtCore.pyqtSignal(int)
    
    roi_x1_changed_signal = PyQt5.QtCore.pyqtSignal(int)
    roi_y1_changed_signal = PyQt5.QtCore.pyqtSignal(int)
    roi_x2_changed_signal = PyQt5.QtCore.pyqtSignal(int)
    roi_y2_changed_signal = PyQt5.QtCore.pyqtSignal(int)
    
    speed_x1_changed_signal = PyQt5.QtCore.pyqtSignal(int)
    speed_y1_changed_signal = PyQt5.QtCore.pyqtSignal(int)
    speed_x2_changed_signal = PyQt5.QtCore.pyqtSignal(int)
    speed_y2_changed_signal = PyQt5.QtCore.pyqtSignal(int)
    
    directory_selected_signal = PyQt5.QtCore.pyqtSignal(str)
    
    change_synth_signal = PyQt5.QtCore.pyqtSignal()
    take_photo_signal = PyQt5.QtCore.pyqtSignal()
    crop_signal = PyQt5.QtCore.pyqtSignal(bool)
    obstacles_off_signal = PyQt5.QtCore.pyqtSignal(bool)
    auto_exposure_signal = PyQt5.QtCore.pyqtSignal(str)
    
    dynamic_expo_signal = PyQt5.QtCore.pyqtSignal(bool)
        
    def __init__(self, parent):
        super().__init__(parent)
        
        self.text_synth_x = PyQt5.QtWidgets.QLineEdit(self)
        self.text_synth_y = PyQt5.QtWidgets.QLineEdit(self)
        self.text_synth_x.setValidator(PyQt5.QtGui.QIntValidator(0, 2000))
        self.text_synth_y.setValidator(PyQt5.QtGui.QIntValidator(0, 2000))
        self.text_synth_x.setText('0')
        self.text_synth_y.setText('0')
        self.text_synth_x.editingFinished.connect(lambda: self.synth_x_changed_signal.emit(int(self.text_synth_x.text())))
        self.text_synth_y.editingFinished.connect(lambda: self.synth_y_changed_signal.emit(int(self.text_synth_y.text())))
        
        # Set Y1 (max) / Y2 (min)
        self.y1_max = PyQt5.QtWidgets.QLineEdit(self)
        self.y2_min = PyQt5.QtWidgets.QLineEdit(self)
        self.y1_max.setValidator(PyQt5.QtGui.QIntValidator(0, 2000))
        self.y2_min.setValidator(PyQt5.QtGui.QIntValidator(0, 2000))
        self.y1_max.setPlaceholderText("MAX DIST")
        self.y2_min.setPlaceholderText("MIN DIST")
        self.y1_max.editingFinished.connect(lambda: self.y1_max_changed_signal.emit(int(self.y1_max.text())))
        self.y2_min.editingFinished.connect(lambda: self.y2_min_changed_signal.emit(int(self.y2_min.text())))
        
        # Save Path
        self.save_path_lineedit = PyQt5.QtWidgets.QLineEdit(self)
        self.save_path_lineedit.setPlaceholderText("Select or enter save path...")
        
        self.save_button = PyQt5.QtWidgets.QPushButton("Browse", self)
        self.save_button.clicked.connect(self.select_save_path)
        
        #  x1, y1, x2, y2 - ROI
        self.text_x1 = PyQt5.QtWidgets.QSlider(PyQt5.QtCore.Qt.Orientation.Horizontal, self)
        self.text_y1 = PyQt5.QtWidgets.QSlider(PyQt5.QtCore.Qt.Orientation.Horizontal, self)
        self.text_x2 = PyQt5.QtWidgets.QSlider(PyQt5.QtCore.Qt.Orientation.Horizontal, self)
        self.text_y2 = PyQt5.QtWidgets.QSlider(PyQt5.QtCore.Qt.Orientation.Horizontal, self)

        self.text_x1.setRange(0, 2448)
        self.text_y1.setRange(0, 2048)
        self.text_x2.setRange(0, 2448)
        self.text_y2.setRange(0, 2048)
        
        self.text_x1.setValue(0)
        self.text_y1.setValue(0)
        self.text_x2.setValue(0)
        self.text_y2.setValue(0)
        
        self.text_x1.valueChanged.connect(lambda v: self.roi_x1_changed_signal.emit(int(v)))
        self.text_y1.valueChanged.connect(lambda v: self.roi_y1_changed_signal.emit(int(v)))
        self.text_x2.valueChanged.connect(lambda v: self.roi_x2_changed_signal.emit(int(v)))
        self.text_y2.valueChanged.connect(lambda v: self.roi_y2_changed_signal.emit(int(v)))
        
        #  x1, y1, x2, y2 - Speed ROI
        self.speed_x1 = PyQt5.QtWidgets.QSlider(PyQt5.QtCore.Qt.Orientation.Horizontal, self)
        self.speed_y1 = PyQt5.QtWidgets.QSlider(PyQt5.QtCore.Qt.Orientation.Horizontal, self)
        self.speed_x2 = PyQt5.QtWidgets.QSlider(PyQt5.QtCore.Qt.Orientation.Horizontal, self)
        self.speed_y2 = PyQt5.QtWidgets.QSlider(PyQt5.QtCore.Qt.Orientation.Horizontal, self)
        
        self.speed_x1.setRange(0, 2448)
        self.speed_y1.setRange(0, 2048)
        self.speed_x2.setRange(0, 2448)
        self.speed_y2.setRange(0, 2048)
        
        self.speed_x1.setValue(0)
        self.speed_y1.setValue(0)
        self.speed_x2.setValue(0)
        self.speed_y2.setValue(0)
        
        self.speed_x1.valueChanged.connect(lambda v: self.speed_x1_changed_signal.emit(int(v)))
        self.speed_y1.valueChanged.connect(lambda v: self.speed_y1_changed_signal.emit(int(v)))
        self.speed_x2.valueChanged.connect(lambda v: self.speed_x2_changed_signal.emit(int(v)))
        self.speed_y2.valueChanged.connect(lambda v: self.speed_y2_changed_signal.emit(int(v)))
                
        self.btn_change_synth = PyQt5.QtWidgets.QPushButton('Change Synth', self)
        self.btn_change_synth.clicked.connect(lambda: self.change_synth_signal.emit())
        
        self.crop_checkbox = PyQt5.QtWidgets.QCheckBox('Image Crop', self)
        self.crop_checkbox.stateChanged.connect(lambda: self.crop_signal.emit(self.crop_checkbox.isChecked()))
        
        self.obstacles_off_checkbox = PyQt5.QtWidgets.QCheckBox('Obstacles off', self)
        self.obstacles_off_checkbox.stateChanged.connect(lambda: self.obstacles_off_signal.emit(self.obstacles_off_checkbox.isChecked()))
       
        self.auto_exposure_text = PyQt5.QtWidgets.QLabel('Auto Exposure', self) 
        self.auto_exposure_text.setAlignment(PyQt5.QtCore.Qt.AlignCenter) 
        
        self.auto_exposure_box = PyQt5.QtWidgets.QComboBox(self)
        self.auto_exposure_box.addItem("OFF")
        self.auto_exposure_box.addItem("CONTINUOUS")
        self.auto_exposure_box.addItem("ONCE")
        self.auto_exposure_box.addItem("CUSTOM")
        self.auto_exposure_box.currentIndexChanged.connect(lambda: self.auto_exposure_signal.emit(self.auto_exposure_box.currentText()))
        
        self.btn_take_photo = PyQt5.QtWidgets.QPushButton('Take photo', self)
        self.btn_take_photo.clicked.connect(lambda: self.take_photo_signal.emit())
        
        self.text_synth_x.hide()
        self.text_synth_y.hide()
        self.btn_change_synth.hide()
        self.crop_checkbox.hide()
        self.obstacles_off_checkbox.hide()
        self.btn_take_photo.hide()
        
        self.speed_x1.hide()
        self.speed_y1.hide()
        self.speed_x2.hide()
        self.speed_y2.hide()
        
        self.my_layout = PyQt5.QtWidgets.QGridLayout()

        self.my_layout.addWidget(PyQt5.QtWidgets.QLabel('X1 (r):'), 4, 0)
        self.my_layout.addWidget(PyQt5.QtWidgets.QLabel('y1 (r):'), 5, 0)
        self.my_layout.addWidget(PyQt5.QtWidgets.QLabel('X2 (r):'), 6, 0)
        self.my_layout.addWidget(PyQt5.QtWidgets.QLabel('y2 (r):'), 7, 0)
        
        self.my_layout.addWidget(self.text_x1, 4, 1)
        self.my_layout.addWidget(self.text_y1, 5, 1)
        self.my_layout.addWidget(self.text_x2, 6, 1)
        self.my_layout.addWidget(self.text_y2, 7, 1)
        
        self.my_layout.addWidget(self.save_path_lineedit, 9, 0)
        self.my_layout.addWidget(self.save_button, 9, 1)
        
        # self.my_layout.addWidget(PyQt5.QtWidgets.QLabel('X1 (s):'), 10, 0)
        # self.my_layout.addWidget(PyQt5.QtWidgets.QLabel('y1 (s):'), 11, 0)
        # self.my_layout.addWidget(PyQt5.QtWidgets.QLabel('X2 (s):'), 12, 0)
        # self.my_layout.addWidget(PyQt5.QtWidgets.QLabel('y2 (s):'), 13, 0)
        
        self.my_layout.addWidget(self.speed_x1, 10, 1)
        self.my_layout.addWidget(self.speed_y1, 11, 1)
        self.my_layout.addWidget(self.speed_x2, 12, 1)
        self.my_layout.addWidget(self.speed_y2, 13, 1)

        self.my_layout.addWidget(self.y1_max, 14, 0)
        self.my_layout.addWidget(self.y2_min, 14, 1)

        self.my_layout.addWidget(self.text_synth_x, 0, 0, 1, 1)
        self.my_layout.addWidget(self.text_synth_y, 0, 1, 1, 1)
        self.my_layout.addWidget(self.btn_change_synth, 1, 0, 1, 1)
        self.my_layout.addWidget(self.btn_take_photo, 2, 0, 1, 1)
        self.my_layout.addWidget(self.crop_checkbox, 1, 1, 1, 1)
        self.my_layout.addWidget(self.obstacles_off_checkbox, 2, 1, 1, 1)
        self.my_layout.addWidget(self.auto_exposure_box, 3, 1, 1, 1)
        self.my_layout.addWidget(self.auto_exposure_text, 3, 0, 1, 1)

        # expanding_size_policy: QSizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        # maximum_size_policy: QSizePolicy = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)

        self.setLayout(self.my_layout)
        self.setFrameShape(PyQt5.QtWidgets.QFrame.Box)
        self.setLineWidth(3)
        
    def get_crop(self):
        return self.crop_checkbox.isChecked()
    
    def get_obstacles_off(self):
        return self.obstacles_off_checkbox.isChecked()
    
    def get_auto_exposure(self):
        return self.auto_exposure_box.currentText()

    def select_save_path(self):
            save_path = PyQt5.QtWidgets.QFileDialog.getExistingDirectory(self)
            
            if save_path:
                self.save_path_lineedit.setText(save_path)
                self.directory_selected_signal.emit(save_path)
    
    def smart_enable(self, status):
        if status == 'READY':
            self.text_synth_x.setEnabled(False)
            self.text_synth_y.setEnabled(False)
            self.btn_change_synth.setEnabled(False)
            
            self.crop_checkbox.setEnabled(True)
            self.obstacles_off_checkbox.setEnabled(True)
            self.auto_exposure_box.setEnabled(True)

            self.btn_take_photo.setEnabled(False)
            
        elif status == 'CAMERA':
            self.text_synth_x.setEnabled(True)
            self.text_synth_y.setEnabled(True)
            self.btn_change_synth.setEnabled(True)
            
            self.crop_checkbox.setEnabled(False)
            self.obstacles_off_checkbox.setEnabled(True)
            self.auto_exposure_box.setEnabled(True)

            self.btn_take_photo.setEnabled(True)
            
        elif status == 'PROCESS':
            self.text_synth_x.setEnabled(True)
            self.text_synth_y.setEnabled(True)
            self.btn_change_synth.setEnabled(True)
            
            self.crop_checkbox.setEnabled(False)
            self.obstacles_off_checkbox.setEnabled(False)
            self.auto_exposure_box.setEnabled(True)

            self.btn_take_photo.setEnabled(True)
            
        else:
            raise ValueError(f'Wrong status {status} !!!')
        
        
########################################################################################################################
