#!/bin/bash
# Exp24 final method on L9 = the real daytime bag (no GT, no first pose). Map = all night laps L0-L8 (test mode).
# Start = stand-in first pose (odo.online --no_first_pose: odo.first_pose, a few m off, sigma 3 m / 1 deg).
# MegaLoc similarity logged on every frame (log_sim, outside the timer).
#   bash scripts/exp24_L9.sh [daytime.mcap]
set -euo pipefail
cd "$(dirname "$0")/.."
BAG=${1:-data/rosbag_2026-10-13_16-08-46_cameras_lap/rosbag_2026-10-13_16-08-46_cameras_lap_0.mcap}
Q=data/L9/extracted
OUT=outputs/24_day/L9/final
mkdir -p logs
[ -f $Q/state.csv ] || python scripts/extract_mcap.py "$BAG" $Q
python -m odo.online --split test --query $Q --out $OUT --no_first_pose \
  --cfg '{"log_sim": true}' 2>&1 | grep --line-buffered -v -i warn | tee logs/exp24_L9.log
