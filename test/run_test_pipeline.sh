#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /23A052/conda/envs/notagen

# Prefer the repo-local abcmidi binary when available
export PATH="/23A052/TUTTI/abcmidi:$PATH"

# Use GPU 7 by default, while exposing it as visible device 0 inside PyTorch
GPU_ID="${GPU_ID:-7}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Run the test pipeline directly
python test_pipeline_new_asap.py --gpu 0 --workers "${WORKERS:-1}"