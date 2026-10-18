#!/bin/bash
# Exp24 final method on night F5/L2, all 7 cameras (map = F2 night laps), for prafful's visualizer
# (scripts/exp24_prafful_viz.py night). Run in tmux, env mast3r-slam.
cd "$(dirname "$0")/.."
python scripts/exp24_day.py final night '{}' --fold F5 --lap L2 2>&1 | grep --line-buffered -E "RESULT|Error|Traceback" | tee -a logs/exp24_viz_night.log
echo NIGHTRUN_DONE | tee -a logs/exp24_viz_night.log
