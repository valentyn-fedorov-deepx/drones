import numpy as np

from src.cv_module.basic_object import BasicObjectWithDistance
from src.cv_module.people.person import Person
from src.cv_module.cars.car import Car


def create_tracked_object_from_detected(detected_object, object_id: int):
    if detected_object.name == "person":
        if detected_object.keypoints is not None:
            kpts=np.c_[detected_object.keypoints, detected_object.keypoints_conf]
        else:
            kpts = None
        return Person(id=object_id,
                      mask=detected_object.mask,
                      bbox=detected_object.bbox,
                      conf=detected_object.conf,
                      timestamp=detected_object.timestamp,
                      kpts=kpts
                      )
    elif detected_object.name in ["car", "truck"]:
        return Car(id=object_id,
                   mask=detected_object.mask,
                   bbox=detected_object.bbox,
                   conf=detected_object.conf,
                   detection=detected_object,
                   timestamp=detected_object.timestamp)
    else:
        return BasicObjectWithDistance(id=object_id,
                                       mask=detected_object.mask,
                                       bbox=detected_object.bbox,
                                       conf=detected_object.conf,
                                       name=detected_object.name,
                                       timestamp=detected_object.timestamp)
