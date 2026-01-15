import os, sys
print("QPA platform =", "xcb")
os.environ["QT_QPA_PLATFORM"] = "xcb"  # X11/VcXsrv
os.environ['PYOPENGL_PLATFORM'] = 'glx'

# важливо: НЕ форси software GL тут
os.environ.pop("QT_OPENGL", None)
os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)
os.environ.pop("MESA_LOADER_DRIVER_OVERRIDE", None)

# не даємо Qt піти в плагіни cv2:
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = (
    f"{sys.prefix}/lib/python{sys.version_info.major}.{sys.version_info.minor}"
    "/site-packages/PyQt6/Qt6/plugins/platforms"
)


import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QProgressBar, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QIcon

from src.gui import back_ui
from src.gui.mywidgest import ImagePane, ReconstructionViewer

class MainWindow(QWidget):

    apply_cloud = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bone Reconstruction GUI")
        self.setGeometry(100, 100, 900, 600)
        
        self.back_end = back_ui.BackUI("http://192.168.4.255:8080/video")
        # self.back_end.status_cb = self.status_callback
        # self.back_end.statusbar_cb = lambda s: self.status_bar.showMessage(s)
        self.back_end.image_cb = self.image_callback
        self.back_end.scene_cb = self.scene_callback
        
        self.back_end.reconstruction_ready.connect(
            self.apply_cloud.emit,
            Qt.ConnectionType.QueuedConnection
        )
        self.apply_cloud.connect(self.scene_callback, Qt.ConnectionType.QueuedConnection)
        self.initUI()
    
    def initUI(self):
        # ==== Upper buttons ====
        self.btn_scan = QPushButton("Scan")
        self.btn_calibrate = QPushButton("Calibrate")

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.btn_scan)
        top_layout.addWidget(self.btn_calibrate)
        top_layout.addStretch()

        # ==== Left Part - CAM ====
        self.image_pane = ImagePane('Camera', self)
        
        self.image_pane.start_signal.connect(self.back_end.start_camera)
        self.image_pane.stop_signal.connect(self.back_end.stop_camera)
        self.image_pane.points_placed.connect(self.back_end.set_points_data)
        self.image_pane.stop_reconstruction_signal.connect(self.back_end.stop_reconstruction)

        # ==== Right Part - 3D ====
        self.reconstruction_viewer = ReconstructionViewer(title="Demo point cloud")

        # ==== Layout CAM / 3D ====
        center_layout = QHBoxLayout()
        center_layout.addWidget(self.image_pane)
        center_layout.addWidget(self.reconstruction_viewer)

        # ==== Lower Buttons ====
        self.btn_capture = QPushButton("capture")
        self.btn_plan = QPushButton("plan")
        self.btn_execute = QPushButton("execute")
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.btn_capture)
        right_layout.addWidget(self.btn_plan)
        right_layout.addWidget(self.btn_execute)
        right_layout.addStretch()
    
        # ==== Progress Bar ====
        self.progress = QProgressBar()
        self.progress.setValue(10)

        # ==== Main Layout ====
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(center_layout)
        main_layout.addLayout(right_layout)
        main_layout.addWidget(self.progress)

        self.setLayout(main_layout)

    def image_callback(self, image: np.ndarray, label_name: str):
        """Update an image label with an OpenCV image"""
        if label_name == 'CAM':
            self.image_pane.set_frame(image)
        else:
            self.image_pane.set_frame(image)

    def scene_callback(self, data):
        print(data[0].shape, data[1].shape)
        points, cols = data
        self.reconstruction_viewer.set_pointcloud(
            points=points,
            colors=cols,
            max_points=200_000,    
            point_size=2.0,         
            px_mode=True,           
            gl_options="opaque",   
            fit=True,              
            margin=1.2              
        )
    
    def closeEvent(self, event):
        try:
            self.back_end.stop_camera()
            self.back_end.proc_thread.quit()
            self.back_end.proc_thread.wait(1000)
        finally:
            event.accept()
        
    

if __name__ == "__main__":
    from src.gui.mywidgest import configure_qt_opengl_desktop_21
    configure_qt_opengl_desktop_21()     

    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())
