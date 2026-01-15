import os
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt
import sys

# paths to append to sys.path for run
cur_dir = Path(__file__).parent.resolve()
data_dir = cur_dir / "data"

reconstr_files_path = data_dir / "reconstr_files_tmp_stream1"

reconstr_stream_path = Path("/nvme0n1-disk/stepan.severylov/sawbone_app/src/thirdparty/StreamVGGT").resolve()
src_stream = Path("/nvme0n1-disk/stepan.severylov/sawbone_app/src/thirdparty/StreamVGGT/src").resolve()
dx_vyzai_path = cur_dir.parent.parent
segment_path = Path("/nvme0n1-disk/stepan.severylov/sawbone_app/thirdparty/sam2").resolve()
code_path = Path('/nvme0n1-disk/stepan.severylov/sawbone_app/src/reconstraction_pipe').resolve()

# adding some paths to sys.path
for p in [reconstr_stream_path, src_stream, dx_vyzai_path, segment_path, code_path]:
    sys.path.append(str(p))

print(reconstr_stream_path.exists())
# libs import
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from streamvggt.utils.geometry import unproject_depth_map_to_point_map

from streamvggt_processor import StreamVGGTProcessor
from frame_loader import FrameLoader
from visualization import plot_pointcloud_plotly, append_pointcloud
from sam2_model import SegmentAnything2, union_binary_masks


ckpt_path = reconstr_stream_path / "ckpt/checkpoints.pth"
proc = StreamVGGTProcessor(
    device="cuda",
    model_path=str(ckpt_path),
    max_keep=64, # live state sliding window
    live_capacity=60_000, # live sampler default value
)
segm = SegmentAnything2()

# set bit_depth and color_data variable to override,
# leave None for inferring from the data
bit_depth = 8
color_data = True
frame_loader = FrameLoader(
    bit_depth,
    color_data
)

#######* FRAME CHOOSING LOGIC ##########
# directory with pxis
pxi_dir = Path("/nvme0n1-disk/volodymyr.danylov/VyzAIProjects/dx_vyzai_python/aux_codes/sawbone_surgical/data/ImprovedSawbones-22-43-34")
pxi_list = sorted(pxi_dir.glob("*.pxi"))

quality_list = frame_loader.select_frames(
    files=pxi_list,
    start=None,
    end=None,
    count=20
)

#! SAVE A FRAME BEFORE PROCESSING - IMPORTANT
image_paths = frame_loader.save_frames(
    paths=quality_list,
    out_dir=reconstr_files_path,
    view="view_img",
    brightening_factor=2
)

#########* Initialize the SAM2 #########
clicks = [
    {"x": 1496, "y": 390},
    {"x": 1205, "y": 1022},
]
init_frame = cv2.cvtColor(cv2.imread(str(image_paths[0])), cv2.COLOR_BGR2RGB)
obj_ids, obj_masks = segm.init_camera(click_coords=clicks, frame=init_frame, return_masks=True)

def function_step(frame_path):
    obj_ids, obj_masks = segm.propagate_binary(
        cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
    )
    union_mask = union_binary_masks(obj_masks)
    frame = proc.preprocess_image(p, valid_mask=union_mask)
    outs = proc.step(
        frame,
        store_device=torch.device("cpu"),
        keep_images=False,
        conf_thresh=0.6  # live-sampler threshold (percentile)
    )
    pts_show, cols_show = proc.get_live_sample()
    return pts_show, cols_show
    
for p in image_paths:
    pts_show, cols_show = function_step(p)
