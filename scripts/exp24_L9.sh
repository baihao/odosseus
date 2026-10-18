#!/bin/bash
# Exp24 final method on L9 = the daytime bag (data/rosbag_2026-10-13_16-08-46_cameras_lap, no GT, no first pose).
# Map = all night laps L0-L8 (test mode). Start = stand-in first pose (scripts/l9_first_pose.py, a few m off) with
# sigma 3 m / 1 deg. MegaLoc similarity logged on every frame (log_sim, outside the timer). Run in tmux, env mast3r-slam.
cd "$(dirname "$0")/.."
Q=data/L9/extracted
OUT=outputs/24_day/L9/final
[ -f $Q/state.csv ] || python scripts/extract_mcap.py data/rosbag_2026-10-13_16-08-46_cameras_lap/rosbag_2026-10-13_16-08-46_cameras_lap_0.mcap $Q
python scripts/l9_first_pose.py $Q 2>&1 | tee logs/exp24_L9.log
python -m odo.online --split test --query $Q --out $OUT \
  --cfg '{"init_sigma_xy": 3.0, "init_sigma_yaw_deg": 1.0, "log_sim": true}' 2>&1 | grep --line-buffered -v -i warn | tee -a logs/exp24_L9.log
echo L9_DONE | tee -a logs/exp24_L9.log
