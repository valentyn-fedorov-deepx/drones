# By Oleksiy Grechnyev, 5/13/24

import os
import sys

# import numpy as np
# import cv2 as cv

# import logging

import PyQt5.QtWidgets
import PyQt5.QtCore
import PyQt5.QtGui

from . import mywidgets


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
class RangeScaleWindow(PyQt5.QtWidgets.QFrame):
    def __init__(self, back_end, parent=None):
        super().__init__(parent)
        self.back_end = back_end

        with open('src/qss/dark.qss', 'r') as f:
            self.setStyleSheet(f.read())

        self.slider_scale = mywidgets.FancySlider('Range Scale', 0.5, 2.0, 1.0, 3, self)
        self.slider_scale.update_value(self.back_end.range_scale)
        self.slider_scale.update_signal.connect(self.back_end.set_range_scale)

        self.my_layout = PyQt5.QtWidgets.QVBoxLayout()
        self.my_layout.addWidget(self.slider_scale)
        self.setLayout(self.my_layout)

        self.setWindowTitle('Range Scale')

########################################################################################################################
