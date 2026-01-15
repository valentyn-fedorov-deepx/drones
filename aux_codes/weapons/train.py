from ultralytics import YOLO
import os

# os.environ['WANDB_MODE'] = 'disabled'

model = YOLO("yolo11m.pt")

model.train(data='/nvme0n1-disk/stepan.severylov/datasets/weapon_types/data.yaml',  
            epochs=100,                         
            batch=32,                          
            imgsz=640,   
            mosaic=0,                               
            name='weapon_training_types_wm')    