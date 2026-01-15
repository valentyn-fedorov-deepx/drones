VyzAI RnD Codebase, DeepX
================
Code originally by 
  > Volodymyr Fedynyak - GUI, People Tracking   
  Oleksiy Hrechniev - GUI, PeopleRangeEstimator(Sloth)   
  Sviatoslav Darmohrai - Was the main maintainer. DataPixelTensor, Camera controllers, Base objects, servers, scripts for data processing   
  Kyryl Dybovetsky - Sawbones   
  Stepan Severylov - OCR   
  Oleksiy Horobets - Webrtc streaming   
  Volodymyr Danytlov - Project S, geolocation tools   
  Anatolii Kashchuk - Project U   
  

Repo house rules
---

Please, if possible:

1. Write good code with at least a minimal amount of docstrings + comments.  
2. Do not commit garbage, outputs, or large files (like neural nets, GroundingDino or output videos), always check `.gitignore`.
3. Do not use windows backslashes in paths anywhere.  
4. Do not commit hardcoded deployment-specific stuff that cannot run on a desktop PC, make it optional. This includes using polarization camera and TensorRT models.
5. Keep separate codes or forked codes in separate **directories**, not git branches. Git branches should be merged to master or deleted within a week or so. Please don't create a repo with multiple de-facto master branches.  
6. Otherwise, create as few forks as possible, better make the main code customizable via config. 

If you don't like these rules, we can discuss it.  
© Oleksiy Grechniev

## Goal of this repository
I've intended for this repository to be as a monorepo that will contain all the python code related to the projects that use sensor with polarization. Why? Because we can reuse a lot of stuff over several projects, and it would be good to have the same interface for this methods. For this DataPixelTensor, DetectedObject, TrackedObjectWithDistance were created as structures that will hold data that can be processed by ProjectManager objects that will be different from project to project and implement the unique business logic. But it proved to not be so sustainable with amount of work we need to do, so some projects, do not fully use this codebase (f.e. Sawbones, Project S). Right now I suggest that this repo can be used to hold python versions of the Camera Controllers, scripts for processing our data, normals calculation. Which are needed for all projects that use the polarization camera. Having this parts working stable it would be much easier to complete urgent requests from the VyzAI team.    

Regarding the import issues that does not cause the script to crash don't worry it is intended this way. So we wouldn't import dependencies from other projects that we do not need in the code we are currently running.   

© Sviatoslav

## Important notes
We assume that you will treat the repo as a python package, and run each file as the separate module in this package. For example:

`python -m deployment.project_la.websocket_server` - Starts websocket server  
`python -m run_scripts.camera.remote_camera_management` - Starts GUI that can connect to the HTTP remote camera management server  

## Important code
As I've written previously the main parts
- **./src/camera** - Camera controllers are here. Right now for IDS and Daheng cameras. Both controllers have the same interface so they can be used interchangably, by specifing the `CAMERA_VENDOR` in the `src/camera/__init__.py` file.
- **./deployment/remote_camera_management** - server that enables camera control over the HTTP API. Can be used to start/stop .pxi data collection
- **./run_scripts/generate_executable** - Here are scripts for generating executables from scripts in our python codebase
- **./run_scripts/gen_frames_from_pxi** - Script for processing the .pxi files. Useful for data processing, where we need to decrease size of the recorded data, and save only relevant bits(RGB images).
- **./src/data_pixel_tensor.py** - DataPixelTensor. Our main data structure is defined here
- **./src/utils/raw_processing** - Here is implementation of demosaicing, normal calculations with different backends(torch, numpy). To change the backend you need to edit DATA_PIXEL_TENSOR_BACKEND variable in the `configs/data_pixel_tensor.py` file.

Regarding the GUI.  
As of now, it just shows input data on the left, and displays processed result on the right. One unfortunate thing is that the ProcessingManager responsible for the generated image is hardcoded in the `src/threads/legacy/thread_processing.py` file. It would be good to have config .yaml file to specify the needed processing manager.



## Repo structure
For more details look inside each folder
- **aux_codes/** - Here should be code for training, evaluating models, forks of some other repos, data gathering, management scripts. Basically, here should be everything that is not used directly in the developed projects, but needed to generate data, models, analysis.
- **configs/** - Configs for different pipelines/projects are stored here
- **deployment/** - Here should be code for deploying our solutions on the servers. 
- **notebooks/** - Notebooks for data analysis, vizualization and quick tests. But it would be better not to commit large notebooks.
- **requirements/** - All requirements should be here 
- **run_scripts/** - Entry point for scripts that run projects/pipelines, run GUIs, utilities(camera stuff, scripts generation of executables).
- **src/** - Main code for running all of the supported projects
- **envs/** - Python environments. Since this repository supports more than one project, each with different dependencies, you will most certainly have more than one 
- **models/** - weights for neural networks used in our pipelines


## Project documentation
Documentation/Scope:

* Project S: [link](https://docs.google.com/document/d/1uvBMR_Bq7gTcBmZetMlvBzAVpDx6elUEGTDVqLzfTo8/edit?tab=t.0) 
* Project AR: [link](https://docs.google.com/document/d/1uvBMR_Bq7gTcBmZetMlvBzAVpDx6elUEGTDVqLzfTo8/edit?tab=t.g3qif8p85isr#heading=h.zdjrq4bgxght)
* Project LA: [link](https://docs.google.com/document/d/1zYvf5DaoM8qyFaxZjCXmzygOCXm4c85K1ChsGk_rCWU/edit?tab=t.0#heading=h.rwzgohk4i5x4)
* Project C: [link](https://docs.google.com/document/d/1jOLU9G6a_1ldPVOla1WjcSaHEESHezMcSh4LQigQtMo/edit?tab=t.0#heading=h.rwzgohk4i5x4)
* Project OCR: [link](https://docs.google.com/document/d/1N_-ZlNOp4ieaVXUXbjIG-77tvPoMy0DxLWQt-0uyLnQ/edit?tab=t.0#heading=h.rwzgohk4i5x4)
* 


## Data structure
Here will be described how we should store data at our servers in more structured way.

First we will split the data from the source type in which it was gathered:
**image_source/** - Images, can be raw data from the our polarized sensor in .tif format, and also just images we took with our phones for testing.
**pxi_source/** - Data that was saved in the custom .pxi format
**video_source/** - Data, that was taken in the video format. For example, raw videos from the old People Track python application

Each of this directories should contain folders that have data for single recording session. Each recording session should contain following items inside it:

One of these are necessary:  
**images/** - data in the pxi format. Files should be numerated in the {video_name}-
.pxi format    
**pxi/** - data in the pxi format. Files should be numerated in the {video_name}-{frame_idx}.pxi format  
**video/** - sources in the video format. Also can be video generated from sources. Here no strict names right now, just make it informable for example, if it is full video, specify it in the name, if part specify start and end frame indices, also specify what elements of DataPixelTensor were used to generate this video.   

And also there should be file with more detailed description about the data.  
**metadata.yaml**  

metadata.yaml structure
```
project_source: project_c | project_la | sawbones | ocr | project_s

location: outdoors | indoors  

daytime: day | night  

movement: static | dynamic  

lens: 50mm | 75mm | etc  

environment: nature | cityscape  

tags: # objects of interest present on the scene  
  - people  

references_dist**: # If we have references to the objects in the scene thay can be specified here (thought it should be more detailed). Should be in meters  
  people: 100  
  car: null  

color_data: False # If data was taken with color sensor  

altitude: # Optional, if we know gps location for the static videos   
longitude: # Optional, if we know gps location for the static videos   
latitude: # Optional, if we know gps location for the static videos   
bearing: # Optional, if we know gps location for the static videos  

pitch: # Optional, rotation angle of the camera for relative to the ground  
roll: # Optional, rotation angle of the camera for relative to the ground  

timestamp: # Optional, if we know one
description: # Optional, some textual description if necessary
```
