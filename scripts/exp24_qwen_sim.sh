#!/bin/bash
# Exp24: F5/L2 with the Qwen day proxy (cl + cr), current defaults + log_sim (MegaLoc similarity logged on every frame
# for the run report; outside the timed loop, poses unchanged). The night run first = grey reference in the report.
# Run in tmux, env mast3r-slam.
cd "$(dirname "$0")/.."
run() { python scripts/exp24_day.py "$@" --fold F5 --lap L2 --cams cl cr 2>&1 | grep --line-buffered -E "RESULT|Error|Traceback" | tee -a logs/exp24_qwen_sim.log; }
run simlog night '{"log_sim": true}'
run simlog day_qwen '{"log_sim": true}'
echo QWENSIM_DONE | tee -a logs/exp24_qwen_sim.log
