#!/bin/bash
# Build the cached 3D feature map of the training run (all 7 cameras). GPU, run inside tmux.
source /scratch/local/velikanov/miniconda/bin/activate mast3r-slam
cd /scratch/local/velikanov/work/odosseus
python -m odo.map_build --cams fr cl cr rr fl r rl --out outputs/20_map
echo MAP_DONE
