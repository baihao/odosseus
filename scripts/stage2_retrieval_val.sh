#!/bin/bash
cd /scratch/local/velikanov/work/odosseus
source /scratch/local/velikanov/miniconda/bin/activate mast3r-slam
python scripts/sweep_retrieval_tune.py dinov2
ALL7="fl fr r rl rr cl cr"
python -m odo.retrieval --desc outputs/10_vpr_desc --model dinov2 --cams $ALL7 --k 10 --out outputs/10_retrieval_dinov2 \
   --name "Exp10 DINOv2-GeM retrieval, 7 cams, top-10" --video
python -m odo.retrieval --desc outputs/10_vpr_desc --model megaloc --cams $ALL7 --k 10 --out outputs/11_retrieval_megaloc \
   --name "Exp11 MegaLoc retrieval, 7 cams, top-10" --video
python -m odo.retrieval --desc outputs/10_vpr_desc --model megaloc --cams fr --k 1 --mode top1 --out outputs/11_retrieval_megaloc/single_fr_top1 \
   --name "MegaLoc retrieval, fr only, top-1"
echo S2_DONE
