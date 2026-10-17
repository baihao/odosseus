#!/bin/bash
# Run MASt3R-SLAM (calibrated) on a prepared sequence; per-frame online poses -> outputs/<exp>/frames_T_WC.txt
# usage: scripts/run_mast3r.sh <prepared_dir> <exp_dir> [config]
set -e
SEQ=$(realpath $1); EXP=$(realpath -m $2); CFG=${3:-config/base.yaml}
mkdir -p $EXP
cd /scratch/local/velikanov/work/odosseus/MASt3R-SLAM
source /scratch/local/velikanov/miniconda/bin/activate mast3r-slam
export TMPDIR=/scratch/local/velikanov/work/odosseus/tmp_pip
/usr/bin/time -v python main.py --dataset $SEQ --config $CFG $( [ -z "$NOCALIB" ] && echo --calib $SEQ/calib.yaml ) --no-viz \
   --frame_log $EXP/frames_T_WC.txt --save-as $(basename $EXP) 2>&1
echo RUN_DONE
