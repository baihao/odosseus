# odosseus: camera-only localisation of the A2RL race car

Causal estimate of the car's `x, y, yaw` (`constructor0/base_link` in `map`) for every 7-camera image set at 10 Hz,
from the images, the camera calibration, the allowed car-state fields (speed, accelerations, yaw rate) and the first
pose of the run.

**Method.** An EKF on (x, y, yaw) predicts with car-state dead reckoning. Every image set is matched to an offline
3D feature map of the training run (SuperPoint keypoints, triangulated with the training GT), guided by the
prediction. One joint 7-camera PnP (with small roll / pitch / height corrections) gives a fix that updates the
filter after a Mahalanobis gate. When no fix is accepted for 1 s, MegaLoc retrieval over the training images proposes
candidates that are verified with the same PnP. Details: [`docs/final_method.md`](docs/final_method.md).

## Results

| evaluation | position RMSE | notes |
|---|---|---|
| Exp21 lap cross-validation, F1–F3 (each of 6 night laps validated once against a map without it) | **0.156 m** | median 0.066 m, lateral 0.131 m, yaw 0.132°, 99.5 % < 1 m ([`docs/cv_performance.md`](docs/cv_performance.md)) |
| Exp24 day transfer, F5/L2 (hardest lap), night→day translated queries vs night map | 0.19–0.20 m | night 0.30 m ([`docs/final_method.md`](docs/final_method.md)) |
| Exp24 day transfer, untouched lap F5/L7 | 0.082 m | night 0.089 m |
| real daytime lap L9 (organizers' daytime sample, no GT) vs night map | – | map fix in 94 % of frames, 33 ms/frame ([`docs/progress_log.md`](docs/progress_log.md)) |

Runtime: ~41 ms per 7-camera image set on one H100 NVL (column `ms` of `online_info.csv`).

## Install

Python 3.11 and one CUDA GPU (developed on an H100 NVL, CUDA 12.4).

```bash
conda create -n odosseus python=3.11 && conda activate odosseus
pip install -r requirements.txt
```

The networks (SuperPoint, MegaLoc) load from the checkpoints' `torch_home/` when it is present; otherwise they
download on first use, so that run needs internet.

## Run on a test bag

```bash
bash scripts/download_checkpoints.sh <URL or folder>       # load + verify the provided checkpoints into outputs/
./run.sh <test.mcap> [out_dir] --first_pose [FILE]          # FILE omitted: the first /tf pose in the test bag
./run.sh <test.mcap> [out_dir] --no_first_pose              # fallback when no first pose is available
```

Output in `<out_dir>` (default `outputs/submission`): `submission.csv` (`timestamp_ns,x,y,yaw`, one row per image
set, header stamp of the image set), `runtime.txt` (poses, 1 s coverage, mean processing time per frame, GPU),
`online_info.csv` (per-frame diagnostics) and `run.log`. A start-pose flag is required; without one, `run.sh` prints
the options and exits. It checks the GPU, the checkpoints and the test bag before the long steps.

* `--first_pose [FILE]`: the given first pose of the run (rules: allowed as the initial pose). FILE is a csv whose
  first row has `x, y, yaw` (yaw in rad, or a quaternion `qx, qy, qz, qw`) and optionally `timestamp_ns` (without it,
  the pose at the first image). Without FILE, the first `/tf` pose of the test bag is used. Right after extraction
  the test run's `/tf` is cut to that first row and its car state to the allowed fields, so no later ground truth
  reaches the localiser.
* `--no_first_pose`: a stand-in start pose. The whole run's car-state motion is matched by curvature to a night lap of
  the map run ([`odo/first_pose.py`](odo/first_pose.py)). It is a few metres off, so the filter starts with
  sigma 3 m / 1°.

Inference reads only the test bag and the checkpoints, not the training bag.

## Checkpoints

The offline map of the training run, generated once and provided with the submission:

| path | size | content |
|---|---|---|
| `outputs/20_map/<cam>.npz` (7 cameras) | 10.6 GB | 3D map: SuperPoint keypoints of every training frame, matched with LightGlue and triangulated with the training GT; 3D points, descriptors, map frame poses |
| `outputs/20_map/train_ref.npz` | small | training GT poses and `pos_accuracy` read at inference (relocalisation database, map filter, stand-in) |
| `outputs/10_vpr_desc/megaloc_*.npz` | 1.4 GB | MegaLoc descriptors of the training images (relocalisation database) |
| `outputs/torch_home/hub/` (optional) | ~1 GB | torch hub cache: SuperPoint, MegaLoc and DINOv2 code and weights as used for the results (offline load) |

Load: `bash scripts/download_checkpoints.sh <URL or folder>` copies or downloads them into `outputs/` and checks them
against `checkpoints.sha256`. `./run.sh` stops with this instruction when one is missing. The training bag is not
needed at inference. How the checkpoints were built (and how to rebuild them): [`docs/precomputed_data.md`](docs/precomputed_data.md).
The checkpoints contain data derived from the challenge dataset ("for use in the National AI Hackathon 2026 only, no
redistribution") and the SuperPoint weights (non-commercial licence), so they are shared privately with the organizers.

## Pretrained models (declared)

| model | use | source | licence |
|---|---|---|---|
| SuperPoint | keypoints and descriptors: map build and every test image set | weights via [cvg/LightGlue](https://github.com/cvg/LightGlue) | Magic Leap, non-commercial |
| LightGlue | matching between training frames (map build only; not run at inference) | [cvg/LightGlue](https://github.com/cvg/LightGlue) | Apache-2.0 |
| MegaLoc (DINOv2 backbone) | descriptors of the training images (checkpoint) and of the test image set when relocalising | [gmberton/MegaLoc](https://github.com/gmberton/MegaLoc) (`torch.hub`) | see the repository |

No external datasets. Code licence: GPL-3.0 ([`LICENSE`](LICENSE)).

## Presentation and visualization

* `Presentation/`: the slide deck, as HTML (`odysseus-presentation.html`) and as LaTeX Beamer
  (`Presentation/latex/`, build instructions in its README). The car image and the videos it embeds are not in the
  repository.
* `visualization/`: the real-time 10 Hz dashboard (overhead track view, lateral and position error, fixes) used for
  the presentation videos. It runs on the SIFT pipeline `localization/` of branch `praffulr_dev` (`camloc`), which is
  not part of this branch, so it does not run here. This branch's own plots and videos come from `odo/report.py`
  (`report_*.pdf`, `traj_*.html`, `debug_*.mp4`) and `odo/run_report.py`.

## Reproduce the reported numbers

Needs the training bag extracted to `data/extracted` (`python scripts/extract_mcap.py <train.mcap> data/extracted`).
The commands per experiment are in [`docs/README.md`](docs/README.md#reproduce); the documentation of all experiments is in
[`docs/`](docs/).
