#!/bin/bash
# GPU-enabled wrapper script for WSL

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:/home/ysabe/personal/Factory6G/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/home/ysabe/personal/Factory6G/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/ysabe/personal/Factory6G/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:/home/ysabe/personal/Factory6G/.venv/lib/python3.10/site-packages/nvidia/cufft/lib:/home/ysabe/personal/Factory6G/.venv/lib/python3.10/site-packages/nvidia/curand/lib:/home/ysabe/personal/Factory6G/.venv/lib/python3.10/site-packages/nvidia/cusolver/lib:/home/ysabe/personal/Factory6G/.venv/lib/python3.10/site-packages/nvidia/cusparse/lib:/home/ysabe/personal/Factory6G/.venv/lib/python3.10/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"

# Activate virtual environment
source /home/ysabe/personal/Factory6G/.venv/bin/activate

# Run the command
exec "$@"
