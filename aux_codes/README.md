Auxiliary codes
========

**car_classification**: Training simple classifier for estimating size of the car: small, medium, large  
**cityscapes_segmnetation**: Code for training ffnnet40S on cityscapes dataset  
**data_generation**: Script for generating labels for car parts segmentation(car body, wheels, windows) with GroundingDINO and SAM    
**data_processing**: Here is just a single script for generating pose labels on new data with pretrained ultralytics model
**improved_detect**: Code for training custom ultralytics object detector. Allows custom augmentations, and using datasets from different sources.  
**kos_mos**: Prepare data and train obstacle instance segmentation to replace GroundingDINO+SAM  
**part_seg**: Code for training the semantic segmentaion models from segmentation_models_pytorch on the body parts segmentation task.  
**SA-1B**: Code for generating dataset for training the FastSAM model from ultralytics.  
**weapons**: Code for training weapons detection models 