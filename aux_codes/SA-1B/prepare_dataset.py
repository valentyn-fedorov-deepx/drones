import random

from sa import SA1BDataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

import onnx
import onnxruntime
from einops import rearrange

import time
from tqdm import tqdm
from utils import show_mask


def calculate_iou(mask_a, mask_b):
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    
    if mask_a.shape != mask_b.shape:
        raise ValueError("Masks must have the same dimensions.")
    
    intersection = np.logical_and(mask_a, mask_b)
    union = np.logical_or(mask_a, mask_b)
    
    area_intersection = intersection.sum()
    area_union = union.sum()
    
    if area_union == 0:
        return 1.0  
    else:
        return area_intersection / area_union
    
def predict_zero_one_range(ort_session, input_image):
    np_input = input_image / 255
    np_input = np_input.astype(np.float32)

    np_input = rearrange(np_input[None], 'b h w c -> b c h w ')
    ort_inputs = {ort_session.get_inputs()[0].name: np_input}
    ort_outs = ort_session.run(None, ort_inputs)
    output = ort_outs[0][0].argmax(0)
    return output

dataset = SA1BDataset("/sdb-disk/vyzai/datasets/sa-1b")
print(f"Number of samples: {len(dataset)}")

model_weights_path = "/nvme0n1-disk/stepan.severylov/SA-1B/ffnet_78s.onnx"

onnx_model = onnx.load(model_weights_path)
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession(model_weights_path,
                                           providers=["CUDAExecutionProvider"])

for i in tqdm(range(len(dataset))):
    try:
        img, mask, class_ids = dataset[i]
        h, w, c = mask.shape
        
        test_image = np.array(img)
        test_image_resized = cv2.resize(test_image, (2048, 1024))

        predicted = predict_zero_one_range(ort_session, test_image_resized).astype(np.uint8)

        vegetation_mask = cv2.resize(predicted, (w, h))
        vegetation_mask[vegetation_mask != 8] = 0
        vegetation_mask[vegetation_mask == 8] = 255    
                
        labels = list()
        for channel in range(c):
            pre_mask = mask[:,:,channel]
            pre_mask[pre_mask > 0] = 255
            
            contours, _ = cv2.findContours(pre_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            polygon = cv2.approxPolyDP(largest_contour, epsilon=0.001 * cv2.arcLength(largest_contour, True), closed=True)
            
            xb, yb, wb, hb = cv2.boundingRect(largest_contour)
            x1, y1, x2, y2 = (xb, yb, xb + wb, yb + hb)
            
            iou_score = calculate_iou(pre_mask[y1:y2,x1:x2], vegetation_mask[y1:y2,x1:x2])
            
            points = list()
            for point in polygon:
                x, y = point[0]
                
                x = float(x / w)
                y = float(y / h)
                
                points.append(x)
                points.append(y)
            
            if iou_score > 0.15:
                index = 1
            else:
                index = 0
            
            my_list = [index] + points
                    
            labels.append(my_list)    
        
        with open(f"/sdb-disk/vyzai/datasets/sa-1b-vegetation/train/labels/img-{i}.txt", "w") as f:
            for label in labels:
                label2str = " ".join(map(str, label))
                f.write(label2str) 
                f.write('\n')
        img.save(f'/sdb-disk/vyzai/datasets/sa-1b-vegetation/train/images/img-{i}.jpg')
        
    except:
        print("problem... next step")