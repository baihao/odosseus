#!/bin/bash
# Exp24 final method on L9 (daytime bag, no GT) with the F5 map = night laps L0 L1 L3 L4 L6 L8 (as F5/L2), instead of
# all night laps. Same start as scripts/exp24_L9.sh (stand-in first pose, sigma 3 m / 1 deg), then prafful's
# dashboard. Run in tmux, env mast3r-slam.
cd "$(dirname "$0")/.."
Q=data/L9/extracted
OUT=outputs/24_day/L9/F5
python -m odo.online --split test --query $Q --fold F5 --out $OUT \
  --cfg '{"init_sigma_xy": 3.0, "init_sigma_yaw_deg": 1.0, "log_sim": true}' 2>&1 | grep --line-buffered -v -i warn | tee logs/exp24_L9_F5.log
python scripts/exp24_prafful_viz.py day $OUT --query $Q --fold F5 2>&1 | grep --line-buffered -v -i 'warn\|findfont' | tee -a logs/exp24_L9_F5.log
echo L9F5_DONE | tee -a logs/exp24_L9_F5.log
