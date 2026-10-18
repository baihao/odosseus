#!/bin/bash
# Exp24: reloc yaw plausibility gate (0.3 deg + 0.1 deg/s) vs current (1 deg + 2 deg/s).
cd "$(dirname "$0")/.."
Y='"reloc_yaw0_deg": 0.3, "reloc_yaw_rate_deg": 0.1'
run() { python scripts/exp24_day.py "$@" 2>&1 | grep --line-buffered -E "RESULT|Error|Traceback" | tee -a logs/exp24_yawgate.log; }
run yawgate night "{$Y}"
run gain16_yawgate day_nolights "{\"aug\": \"gain1.6\", $Y}"
run keep10_yawgate day_nolights "{\"kp_keep\": 0.1, $Y}"
run yawgate day_nolights "{$Y}"
run yawgate day "{$Y}"
run base night '{}' --fold F1 --lap L5
run yawgate night "{$Y}" --fold F1 --lap L5
for k in "30 3 1 1" "30 10 2 3" "30 25 0 5" "30 80 0 20"; do
  python scripts/kidnap_test.py L5 $k "{$Y}" F1 2>&1 | grep --line-buffered -E "KIDNAP|Error|Traceback" | tee -a logs/exp24_yawgate.log
done
echo YAWGATE_DONE | tee -a logs/exp24_yawgate.log
