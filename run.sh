#!/bin/bash
# odosseus: single run script. Raw test bag (+ its first pose) -> submission CSV (timestamp_ns,x,y,yaw).
#
#   ./run.sh <test.mcap> [out_dir] --first_pose [FILE]   # the given first pose; FILE omitted = first /tf of the bag
#   ./run.sh <test.mcap> [out_dir] --no_first_pose       # no first pose: stand-in from the car-state motion
#
# Needs one CUDA GPU, the environment in requirements.txt (python on PATH, or PYTHON=...) and the checkpoints:
#   outputs/20_map/<cam>.npz            3D map: SuperPoint keypoints of the training run, matched with LightGlue and
#                                       triangulated with the training GT (7 cameras)
#   outputs/20_map/train_ref.npz        training GT poses + pos_accuracy read at inference (scripts/build_train_ref.py)
#   outputs/10_vpr_desc/megaloc_*.npz   MegaLoc descriptors of the training images (relocalisation)
#   outputs/torch_home/ (optional)      torch hub cache with the networks; else SuperPoint and MegaLoc download once
# The training bag is not needed. The track borders are not used.
#
# Writes <out_dir> (default outputs/submission):
#   submission.csv      one pose per image set, header stamp of the image set (the submission file)
#   runtime.txt         poses, 1 s coverage, mean processing time per frame, GPU
#   online_info.csv     per-frame diagnostics (fix status, inliers, ms); run.log: the full log
#   extracted/          the test bag's images, calibration, allowed car state and first pose only
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  cat >&2 <<'EOF'
usage: ./run.sh <test.mcap> [out_dir] (--first_pose [FILE] | --no_first_pose)
  a start-pose flag is required:
    --first_pose [FILE]  the given first pose of the run: csv, first row with x, y, yaw [rad] (or qx, qy, qz, qw)
                         and optionally timestamp_ns (without it: the pose at the first image set);
                         without FILE: the first /tf pose of the test bag (only that row is kept)
    --no_first_pose      no first pose given: stand-in from the car-state motion matched to the map run
                         (a few m off; start sigma 3 m / 1 deg)
EOF
  exit 2
}
die() { echo "[run.sh] error: $*" >&2; exit 1; }
log() { echo "[run.sh] $*"; }

TEST_MCAP=""; OUT=""; MODE=""; FP_FILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --first_pose) MODE=first; if [ $# -gt 1 ] && [[ "$2" != --* ]]; then FP_FILE=$2; shift; fi ;;
    --no_first_pose) MODE=none ;;
    -h|--help) usage ;;
    -*) echo "unknown option $1" >&2; usage ;;
    *) if [ -z "$TEST_MCAP" ]; then TEST_MCAP=$1; elif [ -z "$OUT" ]; then OUT=$1; else usage; fi ;;
  esac
  shift
done
[ -n "$TEST_MCAP" ] && [ -n "$MODE" ] || usage
OUT=${OUT:-outputs/submission}
PY=${PYTHON:-python}
CAMS="fl fr cl cr r rl rr"
export OMP_NUM_THREADS=8

# 0. inputs, GPU, checkpoints (fail before the long steps)
[ -f "$TEST_MCAP" ] || die "test bag not found: $TEST_MCAP"
if [ -n "$FP_FILE" ]; then [ -f "$FP_FILE" ] || die "first pose file not found: $FP_FILE"; FP_FILE=$(cd "$(dirname "$FP_FILE")" && pwd)/$(basename "$FP_FILE"); fi
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
  || die "no CUDA GPU visible to $PY (or torch missing): one CUDA GPU and the environment in requirements.txt are needed"
missing=""
for c in $CAMS; do
  for f in outputs/20_map/$c.npz outputs/10_vpr_desc/megaloc_${c}_train.npz; do [ -f $f ] || missing="$missing $f"; done
done
[ -f outputs/20_map/train_ref.npz ] || missing="$missing outputs/20_map/train_ref.npz"
[ -z "$missing" ] || die "checkpoints missing:$missing
  see README, Quick start: steps 1 and 3 (map, MegaLoc descriptors) and 4-5 (train_ref.npz)"
[ -d outputs/torch_home ] && export TORCH_HOME=$PWD/outputs/torch_home
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
for s in val L8; do   # relocalisation database = all descriptor splits present (train 0-897 s, val 897-1120 s, L8)
  [ -f outputs/10_vpr_desc/megaloc_fl_$s.npz ] || log "warning: MegaLoc descriptors of split $s missing: smaller relocalisation database than for the reported results"
done
log "$(date -Is) test bag $TEST_MCAP -> $OUT, start pose: ${MODE/first/given}${FP_FILE:+ ($FP_FILE)}"

# 1. test bag -> images, calibration, car state; its ground truth is cut to the first pose right away
EX=$OUT/extracted
if [ ! -f "$EX/.done" ]; then
  rm -rf "$EX"
  log "extracting the test bag"
  "$PY" scripts/extract_mcap.py "$TEST_MCAP" "$EX"
  head -n 2 "$EX/gt.csv" > "$EX/first_pose_bag.csv"; rm -f "$EX/gt.csv"     # first /tf pose only (rules)
  "$PY" - "$EX" <<'EOF'
import json, sys
import pandas as pd
sys.path.insert(0, '.')
from odo.data import ALLOWED_STATE, CAMS
d = sys.argv[1]
err = []
for c in CAMS:
    if len(pd.read_csv(f'{d}/images/{c}.csv')) == 0:
        err.append(f'no images of camera {c}')
cal = json.load(open(f'{d}/calib.json'))
err += [f'no calibration (camera_info + tf_static) of camera {c}' for c in CAMS if c not in cal or cal[c]['T_base_cam'] is None]
s = pd.read_csv(f'{d}/state.csv')
need = [c for c in ALLOWED_STATE if c != 'timestamp_ns']
err += [f'car state field {c} missing (state_estimation or validation/twist + accel)' for c in need if c not in s]
if err:
    sys.exit('[run.sh] error: test bag incomplete: ' + '; '.join(err))
# keep only the allowed car-state fields (speed, accelerations, yaw rate): no GT pose fields reach the localiser
s[['header.stamp.sec', 'header.stamp.nanosec'] + need].to_csv(f'{d}/state.csv', index=False)
print(f'[run.sh] test bag ok: {len(s)} car-state rows,', ', '.join(f'{c} {len(pd.read_csv(f"{d}/images/{c}.csv"))}' for c in CAMS), 'images')
EOF
  touch "$EX/.done"
fi

# 2. causal online localisation
if [ $MODE = first ]; then
  FP=${FP_FILE:-$EX/first_pose_bag.csv}
  [ "$(wc -l < "$FP")" -ge 2 ] || die "no first pose in $FP (the test bag has no /tf): give --first_pose FILE"
  FLAG=(--first_pose "$FP")
else
  FLAG=(--no_first_pose)
fi
log "localising"
"$PY" -m odo.online --split test --query "$EX" --out "$OUT" "${FLAG[@]}"

# 3. submission + runtime report (brief: average processing time per frame with the hardware)
cp "$OUT/pred.csv" "$OUT/submission.csv"
"$PY" - "$OUT" <<'EOF'
import sys
import numpy as np, pandas as pd, torch
o = sys.argv[1]
p = pd.read_csv(f'{o}/submission.csv'); inf = pd.read_csv(f'{o}/online_info.csv')
t = (p.timestamp_ns.values - p.timestamp_ns.values[0]) / 1e9
cov = 100 * len(np.unique(np.floor(t).astype(int))) / (int(np.floor(t[-1])) + 1)
ok = np.isfinite(p[['x', 'y', 'yaw']].values).all()
fix = 100 * inf.status.isin(['fix', 'reloc', 'reset']).mean()
L = [f'poses                {len(p)} over {t[-1]:.1f} s ({len(p) / max(t[-1], 1e-9):.2f} Hz), all finite: {ok}',
     f'coverage (1 s bins)  {cov:.1f} %' + ('' if cov >= 90 else '   WARNING: below 90 %, not ranked'),
     f'map fix accepted     {fix:.1f} % of frames',
     f'processing time      {inf.ms.iloc[1:].mean():.1f} ms per image set (mean; median {inf.ms.iloc[1:].median():.1f}, p95 {inf.ms.iloc[1:].quantile(.95):.1f})',
     f'hardware             {torch.cuda.get_device_name(0)}, torch {torch.__version__}']
open(f'{o}/runtime.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
EOF
log "submission: $OUT/submission.csv (report: $OUT/runtime.txt)"
