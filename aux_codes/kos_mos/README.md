KOS-MOS
==========

R&D on training instance segmentation (YOLOv8-seg from ultralytics) and optionally also semantic segmentation (FPN from torchvision) to replace GroundingDINO+SAM.

The idea is to process a small dataset through GroundingDINO+SAM as applied in the `static_obstacle_detection.py` in `people_track_python`, 
and to use the detected obstacle as the ground truth for training.

The classes are taken from `static_obstacle_detection.py`: 'hedge', 'obstacle', 'post', 'tree trunk' for instance segmentation, for semantic segmentation add 'background'.
We can change classes later if needed, at the price of a complete re-training.

The resolution for our neural networks should be 608x512, approximately 1/4 of the original size (2448x2048), with width rounded to x32, very small aspect ratio change. Change later if needed.


The codes:
----


`segmentation1.py` : Train semantic segmentation (currently not used)  
`run_dino_sam.py` : Run raw images through GroundingDINO+SAM to label them and create Data collection  
`view_data_collection.py`: Visualize data collection with labels  
`create_split.py`: From data collection, create a train/val split for training YOLOv8-seg  
`train.py`: Train the model on a train/val spilt  
`infer.py`: Run inference on images  
`test.py`: Test a trained model on a category from data collection (ground truth vs prediction segmentation)  


Data management
----

There are 3 levels of data.

1. **Raw image collection** (`raw_images`) is a collection of named directories aka "categories" (e.g. `parking_lot1`) containing raw images, with arbitrary resolution,
aspect rations etc.

2. **Data collection** contains data processed through `run_dino_sam.py`: 608x512 grayscale images, and segmentation labels. They are still divided by named categories.

3. **Dataset splits** Contains dataset splits (train+val each) ready for training YOLOv8-seg, images can be combined from diferent categories.
