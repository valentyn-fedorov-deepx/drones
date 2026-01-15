import numpy as np
from typing import Tuple, Optional

from src.cv_module.basic_object import BasicObjectWithDistance
from src.cv_module.detected_object import DetectedObject


class PersonDetection:
    def __init__(self, bbox: Tuple[float], conf: float,
                 mask: Optional[np.ndarray] = None,
                 pose: Optional[np.ndarray] = np.zeros((17, 2),
                                                       dtype=np.float32),
                 pose_conf: Optional[np.ndarray] = np.zeros((17, ),
                                                            dtype=np.float32)):
        self.bbox = np.array(bbox)
        self.conf = conf
        self.mask = mask
        self.pose = pose
        self.pose_conf = pose_conf
        self.x_pos = (self.bbox[0] + self.bbox[2]) // 2
        self.y_pos = (self.bbox[1] + self.bbox[3]) // 2
        self.bbox_width = self.bbox[2] - self.bbox[0]
        self.bbox_heigth = self.bbox[3] - self.bbox[1]

    def __str__(self):
        return f"Person(conf={self.conf:.2f}, pos={self.x_pos, self.y_pos, self.bbox_width, self.bbox_heigth})"

    def __repr__(self):
        return f"Person(conf={self.conf:.2f}, pos={self.x_pos, self.y_pos, self.bbox_width, self.bbox_heigth})"


class Person(BasicObjectWithDistance):
    def __init__(self, id: int, bbox: np.ndarray,
                 conf: float, mask: Optional[np.ndarray] = None,
                 kpts: np.ndarray = None, moving: bool = False,
                 **kwargs):
        super().__init__(mask=mask, id=id, bbox=np.array(bbox),
                         conf=conf, name="person", **kwargs)
        self.kpts = kpts
        self.pose_info = set()
        self.moving = moving
        self.occluded = False

    @classmethod
    def from_detection(cls, detection: DetectedObject, id: int = 1,
                       **kwargs):
        kpts = np.concatenate([detection.keypoints,
                               detection.keypoints_conf.reshape((-1, 1))],
                              axis=1)
        return cls(id=id, bbox=detection.bbox, conf=detection.conf,
                   kpts=kpts, **kwargs)

    @property
    def pose(self):
        if self.has_pose:
            return self.kpts[:, :2]

    @property
    def pose_conf(self):
        if self.has_pose:
            return self.kpts[:, -1]

    @property
    def has_pose(self):
        if self.kpts is None or self.kpts.sum() == 0:
            return False

        return True

    @property
    def bbox_width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def bbox_heigth(self):
        return self.bbox[3] - self.bbox[1]

    def __str__(self):
        return f"Person(id={self.id}, conf={self.conf:.2f}, dist={self.dist}, pose={self.has_pose}, pos={self.x_pos, self.y_pos, self.bbox_width, self.bbox_heigth})"

    def __repr__(self):
        return f"Person(id={self.id}, conf={self.conf:.2f}, dist={self.dist}, pose={self.has_pose}, pos={self.x_pos, self.y_pos, self.bbox_width, self.bbox_heigth})"
