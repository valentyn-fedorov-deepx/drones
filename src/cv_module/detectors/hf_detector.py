import numpy as np
from omegaconf import OmegaConf
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch
from typing import List
import os

from src.cv_module.people.person import PersonDetection


class HFDetector:
    def __init__(self, config_dir: str, device: str):
        self.device = device
        self.config = OmegaConf.load(os.path.join(config_dir, "hf_detector.yaml"))
        self.processor = AutoImageProcessor.from_pretrained(self.config.model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(self.config.model_name).to(device)

    def predict(self, image: np.ndarray) -> List[PersonDetection]:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.shape[:2]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes,
                                                               threshold=self.config.conf)[0]

        detected_people = list()
        for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
            label = self.model.config.id2label[label_idx.item()]
            if label == "person":
                person = PersonDetection(box.detach().cpu().numpy(), score.item())
                detected_people.append(person)

        return detected_people
