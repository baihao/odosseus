#!/bin/bash
# Exp24 fold F7 (docs/laps.md): map = all night laps L0-L8 (test-run setting), validation = day proxies of L2, final config.
# The night run is a reference only: its frames are in the map (self-matching floor).
cd "$(dirname "$0")/.."
run() { python scripts/exp24_day.py "$@" --fold F7 --lap L2 2>&1 | grep --line-buffered -E "RESULT|Error|Traceback" | tee -a logs/exp24_F7.log; }
run final night "{\"img_root\": \"$PWD/data/extracted\"}"
run final day '{}'
run final day_nolights '{}'
echo F7_DONE | tee -a logs/exp24_F7.log
