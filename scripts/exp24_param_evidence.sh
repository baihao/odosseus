#!/bin/bash
# Evidence for docs/parameters.md: filter consistency (NIS / NEES) on existing runs and car-state drift vs training GT.
cd "$(dirname "$0")/.."
python scripts/filter_consistency.py outputs/24_day/night/rs03/F1_L5 outputs/24_day/night/final/F5_L2 outputs/24_day/night/final/F7_L2 outputs/24_day/night/rs03_clcr/F5_L2 outputs/24_day/day_nolights/rs03_clcr/F5_L2 outputs/24_day/day_qwen/final_clcr/F5_L2 2>&1 | grep -v -i warn | tee logs/filter_consistency.log
python scripts/dr_calibration.py 2>&1 | grep -v -i warn | tee logs/dr_calibration.log
