#!/bin/bash
# Exp24: weak fixes update position only (yaw_min_inl), on top of the reloc yaw gate.
cd "$(dirname "$0")/.."
Y='"reloc_yaw0_deg": 0.3, "reloc_yaw_rate_deg": 0.1'
run() { python scripts/exp24_day.py "$@" 2>&1 | grep --line-buffered -E "RESULT|Error|Traceback" | tee -a logs/exp24_yawinl.log; }
for n in 60 100; do
  run keep10_yg_yi$n day_nolights "{\"kp_keep\": 0.1, $Y, \"yaw_min_inl\": $n}"
  run gain16_yg_yi$n day_nolights "{\"aug\": \"gain1.6\", $Y, \"yaw_min_inl\": $n}"
  run yg_yi$n day_nolights "{$Y, \"yaw_min_inl\": $n}"
  run yg_yi$n night "{$Y, \"yaw_min_inl\": $n}"
  run yg_yi$n night "{$Y, \"yaw_min_inl\": $n}" --fold F1 --lap L5
done
echo YAWINL_DONE | tee -a logs/exp24_yawinl.log
