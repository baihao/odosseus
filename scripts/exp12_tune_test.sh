#!/bin/bash
cd /scratch/local/velikanov/work/odosseus
source /scratch/local/velikanov/miniconda/bin/activate mast3r-slam
export TORCH_HOME=$HOME/.cache/torch
ALL7="fl fr r rl rr cl cr"
[ -f outputs/12_reloc_pnp/tune_retrieval/retrieval.npz ] || python -m odo.retrieval --desc outputs/10_vpr_desc --model megaloc --cams $ALL7 --k 10 --qsplit tune --out outputs/12_reloc_pnp/tune_retrieval --name "MegaLoc 7 cams top-10 (tune lap)"
python -m odo.reloc_pnp --retrieval outputs/12_reloc_pnp/tune_retrieval --qsplit tune --cams ${CAMS:-fr cl cr rr} --limit ${LIMIT:-300} --dt_ms ${DT:-0} --out outputs/12_reloc_pnp/tune_${TAG:-a}
echo T12_DONE
