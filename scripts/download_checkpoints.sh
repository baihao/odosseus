#!/bin/bash
# Load the checkpoints into outputs/ and verify them (sha256, checkpoints.sha256).
#
#   bash scripts/download_checkpoints.sh <SOURCE>     # SOURCE: a URL (HF_TOKEN=... for a gated repo) or a local folder
#
# SOURCE/<path> must hold each <path> listed in checkpoints.sha256 (paths relative to outputs/):
#   20_map/<cam>.npz, 20_map/train_ref.npz     3D map (SuperPoint + LightGlue, triangulated) and training poses
#   10_vpr_desc/megaloc_*.npz                  MegaLoc descriptors of the training images
#   torch_home/hub/...                         optional: SuperPoint, MegaLoc, DINOv2 code + weights (offline load)
# checkpoints.sha256 is written on the machine that has the checkpoints:
#   (cd outputs && find 20_map 10_vpr_desc/megaloc_* torch_home -type f | sort | xargs sha256sum) > checkpoints.sha256
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=${1:-${CKPT_URL:?usage: download_checkpoints.sh <URL or folder> (or set CKPT_URL)}}
AUTH=(); [ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
mkdir -p outputs
while read -r sum path; do
  [ -z "$path" ] && continue
  mkdir -p "outputs/$(dirname "$path")"
  if [ -f "outputs/$path" ] && (cd outputs && echo "$sum  $path" | sha256sum -c --status); then
    echo "ok (present) $path"; continue
  fi
  if [ -d "$SRC" ]; then
    echo "copy $path"; cp "$SRC/$path" "outputs/$path"
  else
    echo "download $path"; curl -fL --retry 3 -C - ${AUTH[@]+"${AUTH[@]}"} -o "outputs/$path" "$SRC/$path"
  fi
done < checkpoints.sha256
(cd outputs && sha256sum -c --quiet ../checkpoints.sha256) && echo "checkpoints ok: outputs/"
