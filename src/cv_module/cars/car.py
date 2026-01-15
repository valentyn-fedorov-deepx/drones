from collections import Counter, deque
import numpy as np
from typing import Optional

from src.cv_module.basic_object import BasicObjectWithDistance
from src.cv_module.detected_object import DetectedObject


class Car(BasicObjectWithDistance):
    def __init__(self, id: int, bbox: np.ndarray,
                 conf: float, detection: DetectedObject,
                 mask: Optional[np.ndarray] = None,
                 type: str = "medium", **kwargs):
        super().__init__(mask=mask, id=id, bbox=bbox,
                         conf=conf, name="car", **kwargs)
        self._type_history = deque(maxlen=7)
        if type is not None:
            self._type_history.append(type)
        self.detection = detection
        self.plate_bbox = None
        self.plate = None
        self.ocr = None

    def set_plate(self, bbox: list):
        self.plate_bbox = bbox

    @property
    def type(self):
        return Counter(self._type_history).most_common(1)[0][0]

    @type.setter
    def type(self, new_type: str):
        self.set_current_type(new_type)

    def set_current_type(self, new_type: str):
        self._type_history.append(new_type)
