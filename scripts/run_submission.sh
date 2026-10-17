#!/bin/bash
# Produce the submission CSV for a run (e.g. the hidden test run) from the raw data.
#
#   bash scripts/run_submission.sh <test_run.mcap> [out_dir]
#
# Steps (1-3 are one-off and skipped when their outputs exist):
#   1. extract the training mcap (images, car state, calibration, track bounds, GT)       -> data/extracted
#   2. map: SuperPoint features of every training frame, triangulated with the training GT -> outputs/20_map/<cam>.npz
#   3. MegaLoc descriptors of the training images (relocalisation when lost)               -> outputs/10_vpr_desc
#   4. extract the test mcap; only its first pose is kept as input (rules)                  -> <out_dir>/extracted
#   5. causal online localisation (odo.online, split test)                                 -> <out_dir>/pred.csv
# Needs one CUDA GPU (developed on an H100 NVL), conda env from environment.txt. GPU jobs: run inside tmux here.
set -euo pipefail
cd "$(dirname "$0")/.."
TEST_MCAP=${1:?usage: run_submission.sh <test.mcap> [out_dir]}
OUT=${2:-outputs/submission}
TRAIN_MCAP=data/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap
CAMS="fl fr cl cr r rl rr"
export OMP_NUM_THREADS=8

[ -f data/extracted/gt.csv ] || python scripts/extract_mcap.py "$TRAIN_MCAP" data/extracted
for c in $CAMS; do
  [ -f outputs/20_map/$c.npz ] || python -m odo.map_build --cams $c --out outputs/20_map
done
for s in train val; do
  [ -f outputs/10_vpr_desc/megaloc_rr_$s.npz ] || python -m odo.vpr --model megaloc --cams $CAMS --splits $s --out outputs/10_vpr_desc
done

mkdir -p "$OUT"
[ -f "$OUT/extracted/state.csv" ] || python scripts/extract_mcap.py "$TEST_MCAP" "$OUT/extracted"
# keep only the first GT pose of the test run (allowed initial pose); the rest of the GT is never read
if [ -f "$OUT/extracted/gt.csv" ] && [ ! -f "$OUT/extracted/first_pose.csv" ]; then
  python - "$OUT/extracted" <<'EOF'
import sys, pandas as pd
d = sys.argv[1]
pd.read_csv(f'{d}/gt.csv', nrows=1)[['timestamp_ns', 'x', 'y', 'yaw']].to_csv(f'{d}/first_pose.csv', index=False)
EOF
fi
python -m odo.online --split test --query "$OUT/extracted" --out "$OUT"
cp "$OUT/pred.csv" "$OUT/submission.csv"
echo "submission: $OUT/submission.csv (timestamp_ns,x,y,yaw); runtime per frame in $OUT/online_info.csv (column ms)"
