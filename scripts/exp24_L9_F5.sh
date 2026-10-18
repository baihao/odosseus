#!/bin/bash
# Exp24 final method on L9 (real daytime bag, no GT) with the F5 map = night laps L0 L1 L3 L4 L6 L8 (as F5/L2), instead
# of all night laps. Same start as scripts/exp24_L9.sh (--no_first_pose, sigma 3 m / 1 deg). Run exp24_L9.sh first:
# it extracts the bag; the racing-line offset table compares both runs.
set -euo pipefail
cd "$(dirname "$0")/.."
Q=data/L9/extracted
OUT=outputs/24_day/L9/F5
mkdir -p logs
python -m odo.online --split test --query $Q --fold F5 --out $OUT --no_first_pose \
  --cfg '{"log_sim": true}' 2>&1 | grep --line-buffered -v -i warn | tee logs/exp24_L9_F5.log
python scripts/l9_racing_offset.py 2>&1 | tee -a logs/exp24_L9_F5.log   # -> outputs/24_day/L9/racing_line_offset.md
