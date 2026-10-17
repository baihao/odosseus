#!/bin/bash
# Final stage-1 evaluation on val with parameters chosen on the tune lap. Run in tmux.
set -x
cd /scratch/local/velikanov/work/odosseus
source /scratch/local/velikanov/miniconda/bin/activate mast3r-slam
ALL7="fl fr cl cr r rl rr"
python -m odo.fuse_vo --steps outputs/04_feature_vo_sp/steps --noise_steps outputs/04_feature_vo_sp/steps_tune \
   --cams $ALL7 --mode robust --c_scale 1.0 --out outputs/04_feature_vo_sp/fuse_val --video \
   --name "Exp04 essential-matrix VO, 7 cams, robust fusion"
P14=$(for c in $ALL7; do echo -n "${c}_p ${c}_e "; done)
python -m odo.fuse_vo --steps outputs/05_planar_vo/merged_val --noise_steps outputs/05_planar_vo/merged_tune \
   --cams $P14 --mode robust --c_scale 1.0 --out outputs/05_planar_vo/fuse_val --video \
   --name "Exp05 planar+essential VO, 7 cams x 2 estimators, robust fusion"
python scripts/exp06_vo_gyro.py
python -m odo.eval_vo --poses outputs/03_pi_long_fr/camera_poses.txt --fmt c2w --seq data/prepared/fr_val \
   --out outputs/03_pi_long_fr/eval --name "Exp03 Pi-Long (Pi3 backbone, no loops), fr camera" --video
echo STAGE1_DONE
