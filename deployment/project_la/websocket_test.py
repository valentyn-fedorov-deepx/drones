import numpy as np

from src.data_pixel_tensor import DataPixelTensor
from src.cv_module.basic_object import BasicObjectWithDistance


detected_object_1 = BasicObjectWithDistance(0, bbox=np.array([0, 0, 50, 100]),
                                            conf=0.5, name="person")

detected_object_1.set_measurement({'dist': 10, 'X': 12, 'Y': 14, 'Z': 10})

detected_object_2 = BasicObjectWithDistance(1, bbox=np.array([0, 0, 50, 100]),
                                            conf=0.5, name="person")
detected_object_2.set_measurement({'dist': 14, 'X': 41, 'Y': 24, 'Z': 14})

detected_object_3 = BasicObjectWithDistance(0, bbox=np.array([0, 0, 50, 100]),
                                            conf=0.5, name="drone")
detected_object_3.set_measurement({'dist': 24, 'X': 63, 'Y': 14, 'Z': 24})

detected_object_4 = BasicObjectWithDistance(0, bbox=np.array([0, 0, 50, 100]),
                                            conf=0.5, name="car")
detected_object_4.set_measurement({'dist': 52, 'X': 12, 'Y': 63, 'Z': 52})

DUMMY_DETECTED_OBJECTS = [detected_object_1, detected_object_2,
                          detected_object_3, detected_object_4]

DUMMY_DATA_TENSOR = DataPixelTensor(np.zeros((2048, 2448, 3), dtype=np.uint8),
                                    for_display=True)
