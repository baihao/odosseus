#!/bin/bash
# Exp24 with the Qwen-Image-Edit day proxy of L2, cameras cl + cr only (scripts/day_translate_qwen.py --cams cl cr ->
# L2_qwen_raw). Localisation uses cl + cr only; night runs with the same 2 cameras are the references.
# Run in tmux, env mast3r-slam.
#   bash scripts/exp24_qwen.sh refs   # references that need no Qwen images (can run during the translation)
#   bash scripts/exp24_qwen.sh        # register the Qwen images onto the night frames, then the day_qwen runs
cd "$(dirname "$0")/.."
OLD='{"reloc_yaw0_deg": 1.0, "reloc_yaw_rate_deg": 2.0, "yaw_min_inl": 0}'
NIGHT7="{\"img_root\": \"$PWD/data/extracted\"}"        # F7: night frames are in the map -> explicit image root
run() { python scripts/exp24_day.py "$@" --cams cl cr 2>&1 | grep --line-buffered -E "RESULT|Error|Traceback" | tee -a logs/exp24_qwen.log; }

if [ "$1" == "refs" ]; then
  run yg_yi60 night '{}' --fold F5 --lap L2                  # F5/L2: L2 not in the map
  run yg_yi60 day_nolights '{}' --fold F5 --lap L2           # CycleGAN proxy, same cameras, for comparison
  run base night "$OLD" --fold F5 --lap L2                   # Exp21 configuration
  run final night "$NIGHT7" --fold F7 --lap L2               # F7: all night laps in the map (self-matching floor)
  echo REFS_DONE | tee -a logs/exp24_qwen.log
  exit
fi

rm -rf data/day_proxy/L2_qwen_raw/images/fl                  # partial frames of an earlier 7-camera start
python scripts/register_proxy.py data/day_proxy/L2_qwen_raw data/day_proxy/L2_qwen 2>&1 | grep --line-buffered -v -i warn | tee logs/register_L2_qwen.log
python scripts/proxy_alignment.py data/day_proxy/L2_qwen data/day_proxy/L2_qwen_raw data/day_proxy/L2_nolights 2>&1 | grep --line-buffered -v -i warn | tee logs/proxy_align_L2.log
run yg_yi60 day_qwen '{}' --fold F5 --lap L2                 # final configuration
run base day_qwen "$OLD" --fold F5 --lap L2                  # Exp21 configuration
run final day_qwen '{}' --fold F7 --lap L2                   # test-run setting
echo QWEN_DONE | tee -a logs/exp24_qwen.log
