#!/bin/bash
# Exp24 with the CAP-VSTNet style-transfer day proxy of L2, cameras cl + cr only (localisation uses cl + cr only).
# Night references with the same cameras: bash scripts/exp24_qwen.sh refs. Run in tmux.
cd "$(dirname "$0")/.."
source /scratch/local/velikanov/miniconda/bin/activate daytrans
python scripts/day_style_transfer.py --lap L2 --cams cl cr --style_dir data/day_proxy/style_refs_raw \
    --out data/day_proxy/L2_style 2>&1 | grep --line-buffered -v -i warn | tee logs/day_L2_style.log
source /scratch/local/velikanov/miniconda/bin/activate mast3r-slam
python scripts/proxy_alignment.py data/day_proxy/L2_style data/day_proxy/L2_nolights 2>&1 | grep --line-buffered -v -i warn | tee logs/proxy_align_L2_style.log
OLD='{"reloc_yaw0_deg": 1.0, "reloc_yaw_rate_deg": 2.0, "yaw_min_inl": 0}'
run() { python scripts/exp24_day.py "$@" --cams cl cr 2>&1 | grep --line-buffered -E "RESULT|Error|Traceback" | tee -a logs/exp24_style.log; }
run yg_yi60 day_style '{}' --fold F5 --lap L2       # final configuration, L2 not in the map
run base day_style "$OLD" --fold F5 --lap L2        # Exp21 configuration
run final day_style '{}' --fold F7 --lap L2         # test-run setting (all night laps in the map)
echo STYLE_DONE | tee -a logs/exp24_style.log
