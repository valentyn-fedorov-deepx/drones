import numpy as np
from datetime import datetime
from typing import Optional, Tuple


class BasicObjectWithDistance:
    def __init__(self, id: int, bbox: Tuple[int], conf: float,
                 height_in_m: Optional[float] = None,
                 width_in_m: Optional[float] = None,
                 mask: Optional[np.ndarray] = None,
                 name: str = "object", timestamp: float = None,
                 velocity = None):
        self.real_height = height_in_m
        self.real_width = width_in_m
        self.meas = None
        self.bbox = bbox
        self.id = id
        self.conf = conf
        self.name = name
        self.mask = mask
        self.velocity = velocity

        self.x_pos = (self.bbox[0] + self.bbox[2]) // 2
        self.y_pos = (self.bbox[1] + self.bbox[3]) // 2

        if timestamp is None:
            self.timestamp = datetime.now().timestamp()
        else:
            self.timestamp = timestamp

    def to_project_la_dict(self):
        data_dict = dict(id=self.id, name=self.name,
                         timestamp=self.timestamp)

        if self.meas is not None:
            location = dict(latitude=self.meas.latitude,
                            longitude=self.meas.longitude,
                            altitude=self.meas.altitude,
                            X=self.meas.X,
                            Y=self.meas.Y,
                            Z=self.meas.Z)
        else:
            location = None

        data_dict["location"] = location

        return data_dict

    def set_measurement(self, meas):
        self.meas = meas

    @property
    def dist(self):
        if self.meas is None:
            return None

        return self.meas["dist"]

    def __repr__(self):
        return f"{self.name.upper()} {self.id}"

    def __str__(self):
        return f"{self.name.upper()} {self.id}"

