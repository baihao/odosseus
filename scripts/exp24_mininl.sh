#!/bin/bash
# Exp24: lower map-fix inlier minimum (min_inl 30 -> 15), F5/L2, cameras cl + cr. References: yg_yi60_clcr (min_inl 30).
# keep10 = 90 % of keypoints dropped (inlier starvation), where the minimum matters most.
cd "$(dirname "$0")/.."
run() { python scripts/exp24_day.py "$@" --cams cl cr 2>&1 | grep --line-buffered -E "RESULT|Error|Traceback" | tee -a logs/exp24_mininl.log; }
run mi15 night '{"min_inl": 15}' --fold F5 --lap L2
run mi15 day_nolights '{"min_inl": 15}' --fold F5 --lap L2
run keep10 night '{"kp_keep": 0.1}' --fold F5 --lap L2
run keep10 day_nolights '{"kp_keep": 0.1}' --fold F5 --lap L2
run keep10_mi15 night '{"kp_keep": 0.1, "min_inl": 15}' --fold F5 --lap L2
run keep10_mi15 day_nolights '{"kp_keep": 0.1, "min_inl": 15}' --fold F5 --lap L2
echo MININL_DONE | tee -a logs/exp24_mininl.log
