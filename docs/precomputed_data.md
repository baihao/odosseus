# Precomputed data ("latents"): what, where, how large

The final system (Exp21, `odo/online.py`) needs two precomputed data sets from the training run. Both are derived
from the training images (and the training GT, for the map), are computed once offline, and are loaded at start-up.
Nothing is precomputed from the query / test run.

| data | location | files | size on disk | loaded into | memory at run time | needed for |
|---|---|---|---|---|---|---|
| **3D feature map** (Exp20) | `outputs/20_map/` | `<cam>.npz`, 7 files | **10.6 GB** (9.9 GiB) | GPU | ~10.4 GB for the full map (test run); ~2/3 of that for a CV fold | every frame (guided matching + PnP) |
| **MegaLoc global descriptors** (Exp11) | `outputs/10_vpr_desc/` | `megaloc_<cam>_{train,val,L8}.npz`, 21 files | **1.4 GB** | GPU | ~1.4 GB (fp16) | relocalisation only (lost state) |
| DINOv2 descriptors (Exp10, not used by Exp21) | `outputs/10_vpr_desc/` | `dinov2_<cam>_{train,val}.npz`, 14 files | 0.24 GB | – | – | stage-2 comparison only |

Both folders are under `outputs/`, which is git-ignored: they are not in the repository and must be rebuilt (commands
below) or shipped separately with the submission.

## 1. 3D feature map — `outputs/20_map/<cam>.npz`
Built by `odo/map_build.py` (`./run.sh` when missing). For each training frame of a camera: the SuperPoint keypoints
that could be triangulated (LightGlue matches to the previous and next frame, training GT poses), stored as
concatenated arrays plus a per-frame index.

| array | shape (per camera) | dtype | bytes / point | content |
|---|---|---|---|---|
| `desc` | (N, 256) | float16 | 512 | SuperPoint descriptor (unit length) |
| `X` | (N, 3) | float32 | 12 | 3D point in the `map` frame [m] |
| `kp` | (N, 2) | float32 | 8 | pixel position in the source frame (raw, distorted); not loaded at run time |
| `stamp`, `stamp_prev`, `stamp_next` | (F,) | int64 | – | frame stamp and its triangulation partners [ns] |
| `offset`, `count` | (F,) | int | – | slice of the point arrays belonging to each frame |
| `T` | (F, 4, 4) | float64 | – | base pose of the frame (training GT at stamp − 9.1 ms) |
| `dt_ns`, `min_base`, `max_kp` | scalar | | | build parameters (−9.1 ms, 1.5 m, 2048) |

The descriptors are 96 % of the size. Per camera (F = frames with a partner):

| camera | frames F | points N | points / frame | file size |
|---|---|---|---|---|
| fl | 11 816 | 1 020 468 | 86 | 0.54 GB |
| fr | 11 816 | 2 859 916 | 242 | 1.52 GB |
| cl | 11 816 | 3 194 149 | 270 | 1.70 GB |
| cr | 11 816 | 3 858 388 | 327 | 2.05 GB |
| r | 11 811 | 2 109 743 | 179 | 1.12 GB |
| rl | 11 816 | 2 719 260 | 230 | 1.45 GB |
| rr | 11 816 | 4 153 390 | 351 | 2.21 GB |
| **total** | | **19 915 314** | | **10.6 GB** |

At start-up, `FeatureMap` keeps only the frames allowed by the split (fold map laps; whole run for the test run) and
whose training GT is not flagged, and moves `X` and `desc` to the GPU (524 B per point, about 10.4 GB for the full
map). Loading takes ~10 s; the per-camera file is read into CPU memory one at a time (peak ~2–3 GB, within the 20 GB
RAM budget).

Rebuild: `for c in fl fr cl cr r rl rr; do python -m odo.map_build --cams $c --out outputs/20_map; done` (one GPU, ~10 min per camera when run alone, ~70 min for all 7).

## 2. MegaLoc descriptors — `outputs/10_vpr_desc/megaloc_<cam>_<split>.npz`
Built by `odo/vpr.py --model megaloc` (MegaLoc = DINOv2-B + SALAD, torch.hub `gmberton/MegaLoc`). One 8448-D
float16 descriptor per image (car body cropped, resized to 322 px height), plus its `stamp_ns`.

| split file | time range | frames per camera | size per camera |
|---|---|---|---|
| `train` | 0–897 s | 8 970 | 152 MB |
| `val` | 897–1120 s | 2 230 | 38 MB |
| `L8` | 1120.7–1194.4 s | 736 | 12 MB |
| **7 cameras, all splits** | | 11 936 | **1.4 GB** |

Used only by `Reloc` (relocalisation after 1 s without a fix): restricted to the map laps of the split and kept on
the GPU (~1.4 GB). Computing descriptors for the query takes ~4.5 ms per image, and only in the lost state.

Rebuild: `python -m odo.vpr --model megaloc --cams fl fr cl cr r rl rr --splits train val L8 --out outputs/10_vpr_desc`
(a few minutes on one GPU).

## Total GPU memory of the running system
About 12 GB for the full map: map ~10.4 GB, MegaLoc descriptors ~1.4 GB, plus SuperPoint and MegaLoc networks and
working buffers (<1 GB, estimated; not measured separately). It should fit on a 16 GB GPU (untested); for smaller GPUs the descriptors could be kept in fp8 / PCA-reduced or
the map split by track sector and streamed.
