#!/bin/bash
# Exp12 final: MegaLoc retrieval (exp11) + geometric re-ranking + 3-DoF multi-camera PnP, params from tune lap.
cd /scratch/local/velikanov/work/odosseus
source /scratch/local/velikanov/miniconda/bin/activate mast3r-slam
export TORCH_HOME=$HOME/.cache/torch
python -m odo.retrieval --desc outputs/10_vpr_desc --model megaloc --cams fl fr r rl rr cl cr --k 10 --out outputs/11_retrieval_megaloc \
   --name "Exp11 MegaLoc retrieval, 7 cams, top-10"
python -m odo.reloc_pnp --retrieval outputs/12_reloc_pnp/tune_retrieval --qsplit tune --cams fr cl cr rr --dt_ms -9.1 \
   --out outputs/12_reloc_pnp/tune_final --name "Exp12 retrieval + PnP (tune lap)"
python -m odo.reloc_pnp --retrieval outputs/11_retrieval_megaloc --qsplit val --cams fr cl cr rr --dt_ms -9.1 \
   --out outputs/12_reloc_pnp/val --name "Exp12 MegaLoc retrieval + re-ranking + 3-DoF multi-camera PnP" --video
echo E12_DONE
