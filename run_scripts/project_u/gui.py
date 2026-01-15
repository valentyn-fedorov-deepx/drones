import sys
import os
import glob
import numpy as np
import open3d as o3d
from PySide6.QtWidgets import (
    QApplication, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QMenu, QGroupBox, QLineEdit, 
    QTextEdit, QDoubleSpinBox, QScrollArea, QFrame, QGridLayout, QTabWidget
)
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QWheelEvent, QImage
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
import cv2
import lightning_fabric
from src.project_managers.project_u_manager import ProjectUManager

class ImageViewer(QGraphicsView):
    region_selected = Signal(QImage)  # Add this signal

    def __init__(self, window="main"):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setMouseTracking(True) 
        self.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.selecting = False
        self.window = window
        
        
        self.selection_rect = None
        self.selection_start = None
        self.temp_start = None
        self.temp_end = None 

        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        # self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self.measurements = []
        self.temp_start = None
        self.measuring = False

    def load_image(self, image_path):
        self.measurements.clear()
        self.temp_start = None
        self.measuring = False

        self.pixmap = QPixmap(image_path)
        self.image_item.setPixmap(self.pixmap)
        self.image_item.setCursor(Qt.CrossCursor)
        self.setSceneRect(self.pixmap.rect())
        self.resetTransform()

    def display_image(self, image):
        if isinstance(image, str):
            self.load_image(image)
        elif isinstance(image, np.ndarray):
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            qimage = QPixmap.fromImage(
                QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            )
            self.pixmap = qimage
            self.image_item.setPixmap(self.pixmap)
            self.image_item.setCursor(Qt.CrossCursor)
            self.setSceneRect(self.pixmap.rect())
            self.resetTransform()
        elif isinstance(image, QImage):
            self.pixmap = QPixmap.fromImage(image)
            self.image_item.setPixmap(self.pixmap)
            self.image_item.setCursor(Qt.CrossCursor)
            self.setSceneRect(self.pixmap.rect())
            self.resetTransform()

    def enable_measurement(self):
        self.measuring = True
        self.temp_start = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.window == "calibration":
                pos = self.mapToScene(event.position().toPoint())
                self.temp_end = None
                self.temp_start = pos
                self.measurements.clear()
                self.viewport().update()
            elif self.window == "main":
                if event.button() == Qt.LeftButton:
                    self.selection_start = self.mapToScene(event.position().toPoint())
                    self.selecting = True
                    self.selection_rect = None
            else:
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.window == "calibration" and self.temp_start is not None:
            pos = self.mapToScene(event.position().toPoint())
            dist = (pos - self.temp_start).manhattanLength()
            self.measurements.append((self.temp_start, pos, dist))
            self.temp_start = None
            self.temp_end = None
            self.measuring = False
            self.viewport().update()
        elif self.window == "main": 
            self.selecting = False
            self.viewport().update()
            selected_image = self.get_selected_region()
            if selected_image:
                self.region_selected.emit(selected_image)  # Emit here
        else:
            super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self.window == "main" and self.selecting:
            current_pos = self.mapToScene(event.position().toPoint())
            rect = QRectF(self.selection_start, current_pos).normalized()
            self.selection_rect = rect
            self.viewport().update()
        elif self.window == "calibration":
            # Draw temporary measurement line
            if self.temp_start is not None:
                self.temp_end = self.mapToScene(event.position().toPoint())
                self.viewport().update()
        else:
            super().mouseMoveEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        clear_action = menu.addAction("Clear Measurements")
        
        action = menu.exec(event.globalPos())
        if action == clear_action:
            self.measurements.clear()
            self.viewport().update()  # trigger repaint to remove lines
            
    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def drawForeground(self, painter, rect):
        if self.selection_rect:
            painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.DashLine))
            painter.drawRect(self.selection_rect)
        if self.window == "calibration" and self.temp_start and self.temp_end:
            pen = QPen(Qt.DashLine)
            pen.setColor(Qt.red)
            painter.setPen(pen)
            painter.drawLine(self.temp_start, self.temp_end)
        else:
            pen = QPen(QColor("red"), 2)
            painter.setPen(pen)
            for p1, p2, dist in self.measurements:
                painter.drawLine(p1, p2)
                mid = (p1 + p2) / 2
                painter.drawText(mid, f"{dist:.1f}px")
    
    def get_selected_region(self):
        if not self.selection_rect or self.pixmap.isNull():
            return None

        rect = self.selection_rect.toRect()
        image = self.pixmap.toImage().copy(rect)
        return image  # Returns a QImage

    def enterEvent(self, event):
        self.setCursor(Qt.CursorShape.CrossCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.unsetCursor()  # Revert to default cursor
        super().leaveEvent(event)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PS reconstruction")
        self.folder_path = ""
        self.image_viewer = ImageViewer()
        self.image_viewer.setCursor(Qt.CursorShape.CrossCursor)
        self.image_viewer.setStyleSheet("border: 2px solid gray;")

        self.pixel_size = 16.67*0.56e-3  # mm/pxl
        self.focal_length = 6.14

        self.thumbnail_viewer = QWidget()
        self.thumbnail_layout = QHBoxLayout()
        self.thumbnail_layout.setContentsMargins(0, 0, 0, 0)
        self.thumbnail_layout.setSpacing(5)
        self.thumbnail_viewer.setLayout(self.thumbnail_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(120)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setWidget(self.thumbnail_viewer)

        # Tabs
        self.fill_tab_groups()

        self.image_viewer.region_selected.connect(self.selection_image.display_image)

        # Select folder controls
        self.select_folder_btn = QPushButton("Select Folder")
        self.select_folder_btn.clicked.connect(self.select_folder)
        self.browse_group = QGroupBox("Data folder")
        self.folder_path_line = QTextEdit()
        self.folder_path_line.setReadOnly(True)
        self.select_folder_btn.setToolTip("Select a folder containing images for processing")
        self.folder_path_line.setFixedHeight(50)
        vbox = QVBoxLayout()
        vbox.addWidget(self.select_folder_btn)
        vbox.addWidget(self.folder_path_line)
        self.browse_group.setFixedWidth(250)
        self.browse_group.setFixedHeight(150)
        self.browse_group.setLayout(vbox)

        

        # 3D Model Viewer
        self.view_3d_group = QGroupBox("3D reconstruction")
        self.get_3d_btn = QPushButton("Get 3D")
        self.get_3d_btn.clicked.connect(self.get_3d_model)
        self.view_3d_btn = QPushButton("View 3D Model")
        self.view_3d_btn.clicked.connect(self.show_3d_model)
        # Add input for camera parameters: focal length, pixel size
        grid_box = QGridLayout()
        
        vbox_3d = QVBoxLayout()
        vbox_3d.addWidget(self.get_3d_btn)
        vbox_3d.addWidget(self.view_3d_btn)
        self.view_3d_group.setLayout(vbox_3d)

        # Measurement
        self.measure_btn = QPushButton("Measure Distance")
        self.measure_btn.clicked.connect(self.start_measurement)
        self.measure_group = QGroupBox("Measurement")
        vbox_measure = QVBoxLayout()
        vbox_measure.addWidget(self.measure_btn)
        self.measure_group.setLayout(vbox_measure)


        btn_layout = QVBoxLayout()
        btn_layout.addWidget(self.browse_group)
        # btn_layout.addWidget(self.calibration_group)
        # btn_layout.addWidget(self.get_3d_btn)
        # btn_layout.addWidget(self.view_3d_btn)
        btn_layout.addWidget(self.view_3d_group)
        btn_layout.addWidget(self.measure_group)
        # btn_layout.addWidget(self.measure_btn)
        
        # btn_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        btn_layout.addStretch() 

        image_layout = QVBoxLayout()
        image_layout.addWidget(self.image_viewer)
        # image_layout.addWidget(self.scroll_area)
        
        image_and_tabs_layout = QHBoxLayout()
        image_and_tabs_layout.addLayout(image_layout)
        image_and_tabs_layout.addWidget(self.tabs)

        image_and_tabs_scroll_layout = QVBoxLayout()
        image_and_tabs_scroll_layout.addLayout(image_and_tabs_layout)
        image_and_tabs_scroll_layout.addWidget(self.scroll_area)

        layout = QHBoxLayout()
        layout.addLayout(btn_layout)
        layout.addLayout(image_and_tabs_scroll_layout)
        self.setLayout(layout)

    def fill_tab_groups(self):
        self.tabs = QTabWidget()
        self.selection_tab = QWidget()
        self.normals_tab = QWidget()
        self.depth_tab = QWidget()
        self.model3d_tab = QWidget()
        self.camera_calibration_tab = QWidget()

        self.tabs.addTab(self.selection_tab, "Selection")
        # self.tabs.addTab(self.normals_tab, "Normals")
        # self.tabs.addTab(self.depth_tab, "Depth")
        # self.tabs.addTab(self.model3d_tab, "3D")
        self.tabs.addTab(self.camera_calibration_tab, "Camera Calibration")

        # Selection tab
        selection_layout = QVBoxLayout()
        self.selection_image = ImageViewer(window="calibration")
        self.selection_image.setFixedHeight(300)
        self.selection_image.setFixedWidth(500)
        self.selection_image.setCursor(Qt.CursorShape.CrossCursor)
        self.selection_image.setStyleSheet("border: 2px solid gray;")
        selection_layout.addWidget(self.selection_image)

        self.selection_label = QLabel("Draw a line to measure distance")
        self.selection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.selection_calculate_btn = QPushButton("Calculate Distance")
        self.selection_calculate_btn.clicked.connect(self.get_real_distance)

        selection_layout.addWidget(self.selection_label)
        selection_layout.addWidget(self.selection_calculate_btn)
        self.selection_tab.setLayout(selection_layout)


        # Normals tab
        normals_layout = QVBoxLayout()
        self.normals_label = QLabel("Normals will be displayed here.")
        self.normals_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        normals_layout.addWidget(self.normals_label)
        self.normals_tab.setLayout(normals_layout)

        # Depth tab
        depth_layout = QVBoxLayout()
        self.depth_label = QLabel("Depth information will be displayed here.")
        self.depth_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        depth_layout.addWidget(self.depth_label)
        self.depth_tab.setLayout(depth_layout)

        # 3D Model tab
        model3d_layout = QVBoxLayout()
        self.model3d_label = QLabel("3D model will be displayed here.")
        self.model3d_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        model3d_layout.addWidget(self.model3d_label)
        self.model3d_tab.setLayout(model3d_layout)

        # Camera Calibration tab
        # self.calibrate_btn = QPushButton("Calibrate Camera")
        # self.calibrate_btn.setEnabled(False)
        # self.calibrate_btn.clicked.connect(self.calibrate)

        self.calibration_group = QGroupBox("2D calibration")
        vbox_calibrate = QVBoxLayout()
        hbox_calibrate = QHBoxLayout()
        self.calibration_distance = QDoubleSpinBox()
        self.calibration_distance.setRange(0, 10000000.0)
        self.calibration_distance.setSingleStep(0.1)
        self.calibration_distance.setValue(10.0)  # Default value in mm
        self.calibration_distance.setSuffix(" mm")

        hbox_calibrate.addWidget(QLabel("True Distance:"))
        hbox_calibrate.addWidget(self.calibration_distance)
        vbox_calibrate.addLayout(hbox_calibrate)

        self.calibration_pxldistance = QDoubleSpinBox()
        self.calibration_pxldistance.setRange(0, 10000000.0)
        self.calibration_pxldistance.setSingleStep(0.1)
        self.calibration_pxldistance.setValue(10.0)  # Default value in mm
        self.calibration_pxldistance.setSuffix(" pxl")
        hbox_calibrate_pxl = QHBoxLayout()
        # self.pixel_distance_btn = QPushButton("Pixel Distance:")
        # self.pixel_distance_btn.clicked.connect(self.get_calibration_distance)
        # self.pixel_distance_btn.setStyleSheet("text-align: left;")

        # hbox_calibrate_pxl.addWidget(self.pixel_distance_btn)
        hbox_calibrate_pxl.addWidget(QLabel("Pixel Distance:"))
        hbox_calibrate_pxl.addWidget(self.calibration_pxldistance)
        vbox_calibrate.addLayout(hbox_calibrate_pxl)
        
        self.calibration_const = QDoubleSpinBox()
        self.calibration_const.setRange(0, 10000000.0)
        self.calibration_const.setSingleStep(0.1)
        self.calibration_const.setValue(10.0)  # Default value in mm
        self.calibration_const.setSuffix(" mm/pxl")
        hbox_calibrate_const = QHBoxLayout()
        hbox_calibrate_const.addWidget(QLabel("Calibration constant:"))
        hbox_calibrate_const.addWidget(self.calibration_const)
        vbox_calibrate.addLayout(hbox_calibrate_const)
        
        self.Fx_lablel = QLabel("Fx:")
        self.Fx_value = QDoubleSpinBox()
        self.Fx_value.setRange(0, 10000000.0)
        self.Fx_value.setSingleStep(0.1)
        self.Fx_value.setValue(6.14)
        self.Fx_value.setSuffix(" mm")
        hbox_Fx = QHBoxLayout()
        hbox_Fx.addWidget(self.Fx_lablel)
        hbox_Fx.addWidget(self.Fx_value)
        vbox_calibrate.addLayout(hbox_Fx)

        self.Fy_label = QLabel("Fy:")
        self.Fy_value = QDoubleSpinBox()
        self.Fy_value.setRange(0, 10000000.0)
        self.Fy_value.setSingleStep(0.1)
        self.Fy_value.setValue(6.14)
        self.Fy_value.setSuffix(" mm")
        hbox_Fy = QHBoxLayout()
        hbox_Fy.addWidget(self.Fy_label)
        hbox_Fy.addWidget(self.Fy_value)
        vbox_calibrate.addLayout(hbox_Fy)

        self.pixel_distance_btn = QPushButton("Calibrate")
        self.pixel_distance_btn.clicked.connect(self.get_calibration_distance)
        vbox_calibrate.addWidget(self.pixel_distance_btn)
        

        self.calibration_group.setLayout(vbox_calibrate)
        self.calibration_group.setFixedWidth(250)

        vbox_calibrate_tab = QVBoxLayout()
        self.calibrate_image = ImageViewer(window="calibration")
        self.calibrate_image.setFixedHeight(300)
        self.calibrate_image.setFixedWidth(500)
        self.calibrate_image.setCursor(Qt.CursorShape.CrossCursor)
        self.calibrate_image.setStyleSheet("border: 2px solid gray;")
        vbox_calibrate_tab.addWidget(self.calibrate_image)
        vbox_calibrate_tab.addWidget(self.calibration_group)
        self.camera_calibration_tab.setLayout(vbox_calibrate_tab)

    def get_real_distance(self):
        if not self.selection_image.pixmap:
            QMessageBox.warning(self, "No Image", "Load an image first.")
            return
        if not self.selection_image.measurements:
            QMessageBox.warning(self, "No Measurement", "Draw a line to measure distance first.")
            return
        # Get the last measurement
        last_measurement = self.selection_image.measurements[-1]
        pixel_distance = last_measurement[2]
        real_distance = self.calibration_const.value() * pixel_distance
        self.selection_label.setText(f"Measured Distance: {real_distance:.2f} mm")
    
    def get_calibration_distance(self):
        if not self.image_viewer.pixmap:
            QMessageBox.warning(self, "No Image", "Load an image first.")
            return
        # Start a measurement to get the pixel distance
        # self.image_viewer.enable_measurement()
        # self.image_viewer.measurements.clear()
        distance = self.calibrate_image.measurements[0][2]
        self.calibration_pxldistance.setValue(distance)
        self.calibration_const.setValue(self.calibration_distance.value() / self.calibration_pxldistance.value())
        
    
    def get_camera_intrinsics(self):
        pixel_size = 0.56e-6 # mm/pxl
        focal_length = 6.14e-3 # mm


    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        # folder = os.path.join(folder,'original_led')
        if folder:
            self.folder_path = folder
            self.images_path = os.path.join(self.folder_path, "black_subtracked")
            self.output_path = os.path.join(self.folder_path, "output")
            image_files = glob.glob(os.path.join(self.images_path,"*.jpg")) + \
                          glob.glob(os.path.join(self.images_path,"*.png"))
            if image_files:
                im_list = []
                for image_file in image_files:
                    image = cv2.imread(image_file)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    im_list.append(image)
                stacked = np.stack(im_list).astype(np.float32)
                avg_img = np.mean(stacked, axis=0).astype(np.uint8)

                # Load calibration image
                image_calibr = cv2.imread(os.path.join(self.folder_path,"laser_led_off.png"))
                image_calibr = cv2.cvtColor(image_calibr, cv2.COLOR_BGR2RGB)
                self.calibrate_image.display_image(image_calibr)
                # Clear thumbnails
                for i in reversed(range(self.thumbnail_layout.count())):
                    widget = self.thumbnail_layout.itemAt(i).widget()
                    if widget:
                        widget.setParent(None)

                # Create thumbnails
                for img_idx, img in enumerate(im_list):
                    qimage = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimage).scaledToHeight(100, Qt.SmoothTransformation)
                    label = QLabel()
                    label.setPixmap(pixmap)
                    label.setCursor(Qt.PointingHandCursor)
                    label.mousePressEvent = lambda event, img=img: self.image_viewer.display_image(img)
                    self.thumbnail_layout.addWidget(label)

                self.image_viewer.display_image(avg_img)
                self.folder_path_line.setText(self.folder_path)
                # self.calibrate_btn.setEnabled(True)
                # self.image_viewer.load_image()
            else:
                QMessageBox.warning(self, "No Images", "No image files found.")

    def start_measurement(self):
        if self.image_viewer.pixmap:
            self.image_viewer.enable_measurement()
        else:
            QMessageBox.warning(self, "No Image", "Load an image first.")

    def get_3d_model(self):
        if not self.folder_path:
            QMessageBox.warning(self, "No Folder", "Select a folder first.")
            return
        if not self.selection_image:
            QMessageBox.warning(self, "Select a region to continue.")
            return
        
        center = self.image_viewer.selection_rect.center().toPoint()
        cx, cy = center.x(), center.y()
        params = {
            "depth_scale": 1.0,  # Assuming depth values are in millimeters (common for sensors)
            "depth_trunc": 50.0,     # Example: Truncate points beyond 5 meters
            "intrinsics": [self.Fx_value.value()/self.pixel_size,self.Fy_value.value()/self.pixel_size,cx,cy]  # fx,fy,cx,cy
        }
        # Create directory with selected region name and save images
        self.region_name = f"region_{cx}_{cy}"
        self.output_path = os.path.join(self.folder_path, "output", self.region_name)
        os.makedirs(self.output_path, exist_ok=True)

        crop_rect = self.image_viewer.selection_rect.toRect()
        image_paths = glob.glob(os.path.join(self.images_path, "*.jpg")) + \
                       glob.glob(os.path.join(self.images_path, "*.png"))
        for image_file in image_paths:
            image = QImage(image_file)
            if crop_rect.right() >= image.width() or crop_rect.bottom() >= image.height():
                print(f"Crop rectangle is outside image bounds: {image_file}")
                continue
            # Get the selected region from the loaded image 
            cropped = image.copy(crop_rect)

            # Save to output path
            filename = os.path.basename(image_file)
            output_file = os.path.join(self.output_path, filename)
            success = cropped.save(output_file)

        PUM = ProjectUManager()
        PUM.process_images(self.output_path,
                             name_card="*.png",
                             save_folder_path=os.path.join(self.output_path,"output"),
                                params=params
                             )
        self.show_3d_model()

    def show_3d_model(self):
        if not self.folder_path:
            QMessageBox.warning(self, "No Folder", "Select a folder first.")
            return

        ply_files = glob.glob(os.path.join(self.output_path,"output","*.ply"))
        if not ply_files:
            QMessageBox.warning(self, "No 3D Model", "No ply files found.")
            return

        pcd = o3d.io.read_point_cloud(ply_files[0])
        # if not mesh.has_vertex_normals():
        #     mesh.compute_vertex_normals()

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Select 3D Points")
        vis.add_geometry(pcd)
        print("Pick points with mouse clicks, then close the window to finish.")
        vis.run()  # user selects points here
        vis.destroy_window()

        picked_ids = vis.get_picked_points()
        if picked_ids:
            vertices = np.asarray(pcd.points)
            picked_points = vertices[picked_ids]
            print("\n✅ Selected 3D Points:")
            for idx, pt in enumerate(picked_points):
                print(f"{idx + 1}: {pt}")
            # Display message with distance between two last points
            if len(picked_points) >= 2:
                dist = np.linalg.norm(picked_points[-1] - picked_points[-2])
                QMessageBox.information(self, "Distance", f"Distance between the two points: {dist:.2f} mm")
        else:
            print("❌ No points selected.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1500, 800)
    window.show()
    sys.exit(app.exec())
