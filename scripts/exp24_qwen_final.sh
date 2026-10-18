#!/bin/bash
# Exp24: Qwen day proxy of L2 (cl + cr), F5 (map = F2 night laps), current defaults. Run in tmux, env mast3r-slam.
cd "$(dirname "$0")/.."
python scripts/exp24_day.py final day_qwen '{}' --fold F5 --lap L2 --cams cl cr 2>&1 | grep --line-buffered -E "RESULT|Error|Traceback" | tee -a logs/exp24_qwen_final.log
python scripts/exp24_matchdiag.py outputs/24_day/day_qwen/final_clcr/F5_L2 2>&1 | tee -a logs/exp24_qwen_final.log
echo QWENFINAL_DONE | tee -a logs/exp24_qwen_final.log
