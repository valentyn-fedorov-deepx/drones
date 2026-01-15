from typing import Optional
import math

import numpy as np

from src.utils.gps import local_to_gps

CAMERA_MEASUREMENTS_NAMES = {"X", "Y", "Z", "dist"}
GPS_NAMES = {"longitude", "latitude", "altitude", "heading"}
ALL_NAMES = GPS_NAMES.union(CAMERA_MEASUREMENTS_NAMES)


class Measurement:
    def __init__(self, X: float, Y: float, Z: float,
                 velocity: Optional[np.ndarray] = None,
                 longitude: Optional[float] = None,
                 latitude: Optional[float] = None,
                 altitude: Optional[float] = None):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.dist = math.sqrt(self.X**2 + self.Y**2 + self.Z**2)

        self.velocity = velocity
        self.longitude = longitude
        self.latitude = latitude
        self.altitude = altitude
        self.timestamp = None

    @property
    def dist_property(self):
        return math.sqrt(self.X**2 + self.Y**2 + self.Z**2)

    def __getitem__(self, item_name: str):
        if item_name in ALL_NAMES:
            return getattr(self, item_name)
        return None

    def estimate_gps_location(self, cam_latitude, cam_longitude,
                              cam_heading):
        lat, lon = local_to_gps(cam_latitude, cam_longitude, 
                                cam_heading, self.X, self.Z)
        self.latitude = lat
        self.longitude = lon

    def __retr__(self):
        return f"Camera: (X={self.X}, Y={self.Y}, Z={self.Z}), GPS(lon={self.longitude}, lat={self.latitude})"

    def __str__(self):
        return f"Camera: (X={self.X}, Y={self.Y}, Z={self.Z}), GPS(lon={self.longitude}, lat={self.latitude})"
