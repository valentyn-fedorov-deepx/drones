from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel, QMessageBox
from numpy import ndarray

message_types: dict = {
    "about": QMessageBox.about,
    "critical": QMessageBox.critical,
    "information": QMessageBox.information,
    "question": QMessageBox.question,
    "warning": QMessageBox.warning
}


def convert_cv_qt(rgb_image: ndarray, image_label: QLabel) -> QPixmap:
    """Convert from an opencv image to QPixmap"""
    height, width, channel = rgb_image.shape
    bytes_per_line: int = channel * width
    convert_to_qt_format: QImage = QImage(
        rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
    )
    qt_scaled_image: QImage = convert_to_qt_format.scaled(
        image_label.width(), image_label.height(), Qt.AspectRatioMode.KeepAspectRatio
    )
    return QPixmap.fromImage(qt_scaled_image)
