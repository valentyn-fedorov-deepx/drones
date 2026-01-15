import os
import cv2
import numpy as np
from pathlib import Path
from segment_anything import SamPredictor, sam_model_registry

class SAMSegmenter:
    """Segments an image via SAM click simulation."""
    def __init__(self, checkpoint: str, model_type: str='vit_l', device: str='cuda'):
        if not Path(checkpoint).exists():
            url = f"https://dl.fbaipublicfiles.com/segment_anything/{checkpoint}"
            os.system(f"wget -q {url} -O {checkpoint}")
        self.model = sam_model_registry[model_type](checkpoint=checkpoint).to(device).eval()
        self.predictor = SamPredictor(self.model)

    def segment(self, img: np.ndarray, point: tuple) -> np.ndarray:
        self.predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        m,_,_ = self.predictor.predict(
            point_coords=np.array([point],dtype=float),
            point_labels=np.array([1],dtype=int),
            multimask_output=False
        )
        return m[0].astype(bool)