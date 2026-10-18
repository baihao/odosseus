#!/bin/bash
# Exp24: the Qwen day proxy fails at stage-0 matching (median 28 descriptor matches vs 532 at night; logs/exp24_matchdiag.log).
# Looser descriptor threshold (sim1_lo) only when the strict one gives almost nothing (lo_below 30 / 60 instead of 300):
# night L2 has >= 30 strict matches in every frame, so night stays untouched with lo_below 30. Run in tmux, env mast3r-slam.
cd "$(dirname "$0")/.."
run() { python scripts/exp24_day.py "$@" --fold F5 --lap L2 --cams cl cr 2>&1 | grep --line-buffered -E "RESULT|Error|Traceback" | tee -a logs/exp24_lo.log; }
for c in "lo65b30 0.65 30" "lo60b30 0.6 30" "lo65b60 0.65 60" "lo55b30 0.55 30"; do
  set -- $c
  for im in day_qwen night day_nolights; do run $1 $im "{\"sim1_lo\": $2, \"lo_below\": $3}"; done
done
python scripts/exp24_matchdiag.py outputs/24_day/*/lo*b*_clcr/F5_L2 2>&1 | tee -a logs/exp24_lo.log
echo LO_DONE | tee -a logs/exp24_lo.log
