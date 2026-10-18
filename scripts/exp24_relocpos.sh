#!/bin/bash
# Exp24: the Qwen day proxy (F5/L2, cl + cr) failed by one wrong relocalisation, a 21 m jump after 5.7 s without a fix
# (80 inliers, all from cr), accepted because the position plausibility grows by 2 m/s (dead reckoning drifts ~0.06 m/s).
# Test: reloc_sig_rate 2 -> 0.3 m/s; and a looser descriptor threshold when matches are few (sim1_lo).
# Then check the new rule where it could hurt: night, CycleGAN day, stress (7 cameras), L5, kidnaps. Run in tmux.
cd "$(dirname "$0")/.."
RS='"reloc_sig_rate": 0.3'
run() { python scripts/exp24_day.py "$@" 2>&1 | grep --line-buffered -E "RESULT|Error|Traceback" | tee -a logs/exp24_relocpos.log; }
run rs03 night "{$RS}" --fold F5 --lap L2 --cams cl cr
run rs03 day_qwen "{$RS}" --fold F5 --lap L2 --cams cl cr
run lo65 night '{"sim1_lo": 0.65}' --fold F5 --lap L2 --cams cl cr
run lo65 day_qwen '{"sim1_lo": 0.65}' --fold F5 --lap L2 --cams cl cr
run rs03 day_nolights "{$RS}" --fold F5 --lap L2 --cams cl cr
run rs03 day_qwen "{$RS}" --fold F7 --lap L2 --cams cl cr
run keep10_rs03 day_nolights "{\"kp_keep\": 0.1, $RS}" --fold F5 --lap L2
run gain16_rs03 day_nolights "{\"aug\": \"gain1.6\", $RS}" --fold F5 --lap L2
run rs03 night "{$RS}" --fold F1 --lap L5
C="{\"reloc_yaw0_deg\": 0.3, \"reloc_yaw_rate_deg\": 0.1, \"yaw_min_inl\": 60, $RS}"
for k in "30 3 1 1" "30 0 3 0" "30 10 2 3" "30 25 0 5" "30 80 0 20"; do
  python scripts/kidnap_test.py L5 $k "$C" F1 2>&1 | grep --line-buffered -E "KIDNAP|Error|Traceback" | tee -a logs/exp24_relocpos.log
done
echo RELOCPOS_DONE | tee -a logs/exp24_relocpos.log
