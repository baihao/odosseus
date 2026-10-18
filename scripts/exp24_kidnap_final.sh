#!/bin/bash
# Kidnap recovery with the Exp24 configuration (reloc yaw gate + position-only weak fixes), F1 / L5.
cd "$(dirname "$0")/.."
mkdir -p logs
C='{"reloc_yaw0_deg": 0.3, "reloc_yaw_rate_deg": 0.1, "yaw_min_inl": 60}'
for k in "30 3 1 1" "30 0 3 0" "30 10 2 3" "30 25 0 5" "30 80 0 20"; do
  python scripts/kidnap_test.py L5 $k "$C" F1 2>&1 | grep --line-buffered -E "KIDNAP|Error|Traceback" | tee -a logs/exp24_kidnap_final.log
done
echo KIDNAP_DONE | tee -a logs/exp24_kidnap_final.log
