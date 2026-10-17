#!/bin/bash
# Run VGGT-Long / Pi-Long on a prepared sequence; camera_poses.txt (c2w per frame) -> <exp_dir>
# usage: scripts/run_vggtlong.sh <prepared_dir> <exp_dir> <config>
set -e
SEQ=$(realpath $1); EXP=$(realpath -m $2); CFG=$3
mkdir -p $EXP
cd /scratch/local/velikanov/work/odosseus/VGGT-Long
source /scratch/local/velikanov/miniconda/bin/activate mast3r-slam; source .venv/bin/activate
export TMPDIR=/scratch/local/velikanov/work/odosseus/tmp_pip TORCH_HOME=$HOME/.cache/torch HF_HOME=$HOME/.cache/huggingface
rm -rf exps/_odo_tmp; mkdir -p exps
/usr/bin/time -v python vggt_long.py --image_dir $SEQ --config $CFG 2>&1 | grep -v "it/s\]" 
D=$(ls -td exps/*/* | head -1)
cp $D/camera_poses.txt $D/*.yaml $EXP/ ; cp $D/sim3_opt_result.png $EXP/ 2>/dev/null || true
echo "exp dir $D"
echo RUN_DONE
