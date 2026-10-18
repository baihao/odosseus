#!/bin/bash
# Exp24 stress tests on the day proxy without lights (F5/L2): inlier starvation and exposure.
cd "$(dirname "$0")/.."
mkdir -p logs
run() { python scripts/exp24_day.py "$1" "$2" "$3" 2>&1 | grep --line-buffered -E "RESULT|Error|Traceback" | tee -a logs/exp24_stress.log; }
run keep50 day_nolights '{"kp_keep": 0.5}'
run keep25 day_nolights '{"kp_keep": 0.25}'
run keep10 day_nolights '{"kp_keep": 0.1}'
run gain16 day_nolights '{"aug": "gain1.6"}'
run gamma18 day_nolights '{"aug": "gamma1.8"}'
echo STRESS_DONE | tee -a logs/exp24_stress.log
