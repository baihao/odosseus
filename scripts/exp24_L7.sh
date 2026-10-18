#!/bin/bash
# Exp24 check on an untouched lap: F5/L7 (validation lap), Exp21 config vs final, night vs day_nolights proxy.
cd "$(dirname "$0")/.."
mkdir -p logs
OLD='{"reloc_yaw0_deg": 1.0, "reloc_yaw_rate_deg": 2.0, "yaw_min_inl": 0}'
run() { python scripts/exp24_day.py "$@" --lap L7 2>&1 | grep --line-buffered -E "RESULT|Error|Traceback" | tee -a logs/exp24_L7.log; }
run base night "$OLD"; run yg_yi60 night '{}'; run base day_nolights "$OLD"; run yg_yi60 day_nolights '{}'
echo L7_DONE | tee -a logs/exp24_L7.log
