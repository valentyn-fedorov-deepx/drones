#!/bin/bash
cd /mnt/larger_disk/dx_vyzai_python
export PYTHONPATH=${PWD}
export LOGURU_LEVEL="INFO"
/mnt/larger_disk/dx_vyzai_python/.venv/bin/python /mnt/larger_disk/dx_vyzai_python/deployment/webrtc_streaming/server.py