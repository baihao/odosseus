# Camera-based localisation on a race track

**Final method (Exp21), 3-fold lap cross-validation (F1–F3, 6 laps each validated once): 0.156 m position RMSE,
median 0.066 m, 0.131 m lateral, 0.132° yaw, 99.5 % within 1 m; ~41 ms per 7-camera image set** (H100 NVL). Without
the one off-line warm-up lap (L2) the laps are 0.07–0.12 m. Details: [`laps.md`](laps.md),
`outputs/21_online/cv/README.md`, per fold and lap: [`cv_performance.md`](cv_performance.md). Submission for a new run: `bash scripts/run_submission.sh <run.mcap>`.

**Daylight transfer (Exp24, [`final_method.md`](final_method.md), `outputs/24_day/summary.md`).** On night→day
translated query images matched against the night map: F5/L2 (hardest lap) 0.19–0.20 m vs 0.30 m at night, untouched
lap F5/L7 0.082 m vs 0.089 m. Two filter rules added (reloc yaw plausibility, position-only weak fixes) keep the
system stable down to 10 % of the features and under overexposure. They are now the defaults.

Task: estimate the x, y and yaw of the race car (`constructor0/base_link` in `map`) from its 7 cameras.
Data: one night session at Yas Marina, 1194 s long. The score is position RMSE (see the dataset README and
`tools/evaluate.py`). Progress log: [`progress_log.md`](progress_log.md). Detailed explanation of all experiments, with open questions for the
dataset provider: [`experiments_explained.md`](experiments_explained.md). Each experiment has its own README in `outputs/<exp>/`.

## Layout

| path | content |
|---|---|
| `odo/` | the code (Python package, env `mast3r-slam`) |
| `odo/data.py` | access to GT, car state (allowed fields only), calibration, frames; splits |
| `odo/metrics.py` | official metric (bit-exact with `tools/evaluate.py`), plus RPE and aligned ATE |
| `odo/report.py` | per-experiment report: `metrics_<split>.json`, `report_<split>.pdf`, interactive `traj_<split>.html`, `debug_<split>.mp4` |
| `odo/feature_vo.py`, `odo/planar_vo.py`, `odo/fuse_vo.py` | stage 1: the visual odometry we chose |
| `odo/vo2map.py`, `odo/eval_vo.py` | evaluation of third-party VO (VGGT-Long, Pi-Long, MASt3R-SLAM) |
| `odo/vo_gyro.py` | VO combined with the gyro (VO used as a motion model) |
| `odo/vpr.py`, `odo/retrieval.py`, `odo/reloc_pnp.py` | stage 2: retrieval, then metric relocalisation |
| `odo/map_build.py` | stage 3: cached 3D feature map of the training run (SuperPoint + triangulation with train GT) |
| `odo/run_report.py` | per-run PDF: error, fix − dead-reckoning innovation, fix and retrieval rates |
| `scripts/day_translate.py` | Exp24 day proxy of the query images (CycleGAN-Turbo night_to_day, env `daytrans`) |
| `odo/online.py` | stage 4, **final system**: car-state EKF + guided matching to the map + 6-DoF PnP + relocalisation; also test mode |
| `odo/borders.py` | track-border edge alignment (evaluated, not used: too weak with these camera heights) |
| `scripts/run_submission.sh` | single run script: raw mcap of a run → `submission.csv` |
| `scripts/` | data extraction, experiment runners, parameter sweeps, summary reports |
| `outputs/NN_*` | one folder per experiment (README, PDF report, HTML, video, predictions) |
| `outputs/stage{1,2,3}_summary/` | method comparison: `summary.pdf`, `overlay.html`, `metrics.csv` |
| `outputs/20_map/`, `outputs/10_vpr_desc/` | precomputed map (10.6 GB) and MegaLoc descriptors (1.4 GB), git-ignored; see [`precomputed_data.md`](precomputed_data.md) |

## Protocol

* **Laps and split (current protocol: 3-fold cross-validation).** The run has 9 laps L0–L8 (start-line crossings;
  table in [`laps.md`](laps.md), definitions in `odo/data.py: LAPS, FOLDS`).

  | fold | map | laps in the map (training) | validation laps | tuning lap |
  |---|---|---|---|---|
  | F1 | `map_cv_f1` | L0 L2 L3 L4 L7 L8 | L1, L6 | L5 |
  | F2 | `map_cv_f2` | L0 L1 L3 L4 L6 L8 | L2, L7 | L5 |
  | F3 | `map_cv_f3` | L0 L1 L2 L6 L7 L8 | L3, L4 | L5 |

  L5 is always the tuning lap (the only lap used to choose parameters, never in a map). Each of L1, L2, L3, L4, L6, L7
  is validated once; results are reported per lap, per fold and pooled over the 6 laps. Every lap is run as its own
  segment, starting from its first GT pose. For the hidden test run the map is the whole training run.
* **Legacy split (stages 1–2 and the first Exp21 numbers).** train = 0–897 s, tune lap = 632–752 s (≈ L4), dev2 =
  752–897 s (≈ L5), val = 897–1120 s (≈ L6 + L7); for a development lap the map was train minus that lap ±10 s.
  Exp21 parameters were first chosen on tune + dev2. L4 and L5 were therefore seen during development, which matters
  for F3 (L4 is one of its validation laps); see `outputs/21_online/cv/README.md`.
* **Scope of `evaluate.py` runs.** Every prediction file covers only its own segment: a lap (CV) or the legacy val
  window (t = 897–1120 s, 2230 image sets). No method was run over the
  whole 1194 s recording (training laps are the map, so scoring them would be meaningless). `tools/evaluate.py` scores
  every submitted stamp that has GT around it (gap ≤ 50 ms), but computes **coverage over the whole time span of the
  GT file it is given**. With the full `gt.csv` a val-only prediction therefore shows coverage 18.7 % (223 of 1194 s)
  while all other metrics are the same. For the official check we pass GT cut to the val window
  (`outputs/21_online/val/gt_val.csv`, `/tf` rows with 897 s ≤ t < 1120 s):
  `python data/c0_2026-10-13_22-53-42_camloc/tools/evaluate.py outputs/21_online/val/gt_val.csv outputs/21_online/val/pred.csv`
  → 0.097 m, coverage 100 %. For the CV laps the GT cut to each lap is `outputs/21_online/cv/<fold>/<lap>/gt_<lap>.csv`.
  `odo/metrics.py` reproduces the scorer and computes coverage over the split window.
* **Anchoring.** Odometry starts from the GT pose of the first frame of the evaluated segment (lap). The rules allow the first pose of the run;
  for a segment that starts mid-run, this is the equivalent.
* **Allowed inputs.** Images, intrinsics and extrinsics, track borders. From the car state: v, vx, vy, wz, ax, ay
  (Exp21 uses vx, vy, wz, v). The training-run GT may serve as a map (slide 17), including its `pos_accuracy` field to
  skip badly geo-referenced map frames. No method uses GT from the query time; test mode reads only `first_pose.csv`.
* **Causality.** VO step k uses only frames k−1 and k. Retrieval and PnP use only the current image set plus the map.
  The 7 cameras are hardware-synchronised (stamps within 0.02 ms).

## Visualisation (for debugging)

Each experiment has:
1. `report_<split>.pdf`. Page 1: metrics table with baselines. Page 2: map, with the prediction coloured by
   position error. Page 3: position, lateral, longitudinal and yaw error over time (in the GT car frame). Page 4: error CDF.
2. `traj_<split>.html`. Interactive (plotly) map with the track borders, GT and prediction. Hovering a point shows t and its
   lat/lon/yaw errors. A linked plot shows error over time. This was the most useful view for finding where and why
   a method fails.
3. `debug_<split>.mp4`. The fl camera next to a heading-up local map (GT in grey, prediction in blue) with live errors,
   for checking that failures line up with what the camera sees.

## Stage 1: visual odometry (val, 2 laps, 223 s)

| method | pos RMSE | yaw RMSE | RPE 1 s yaw | RPE 10 s drift / yaw |
|---|---|---|---|---|
| dead reckoning from the car state (reference, not VO) | 20.5 m | 2.58° | 0.087° | 0.57 % / 0.36° |
| 02 MASt3R-SLAM | loses tracking after 1 to 3 frames | | | |
| 03 Pi-Long (VGGT-Long with Pi3 backbone), fr camera | 313 m | 26.3° | 1.57° | 4.6 % / 6.2° |
| 04 SuperPoint+LightGlue, essential matrix, 7 cameras, robust fusion | 50.9 m | 5.4° | 0.79° | 2.0 % / 2.5° |
| **05 planar 1-DoF + essential, 7 cameras × 2 estimators** | **43.3 m** | **3.1°** | **0.34°** | **1.25 % / 1.26°** |
| 06 gyro with its bias tracked from VO | 20.6 m | 2.7° | 0.087° | 0.57 % / 0.37° |

Chosen camera-only VO: **Exp05**, a planar multi-camera VO with an Ackermann constraint and metric steps. For each camera
and frame pair, a single yaw θ is solved robustly. The step length comes from the car speed and the extrinsics are known.
The planar θ and the classic essential-matrix rotation are both fused across all 7 cameras with per-camera
gain/bias/σ calibrated on the tune lap. Learned feed-forward systems (MASt3R-SLAM, VGGT-Long, Pi-Long) do not
cope with 3 to 4 m between frames, night scenes and low overlap. The car-state yaw rate is ~4× more accurate than any VO
over 1 to 10 s, and VO did not improve on it (Exp06). Odometry alone drifts to ~20 m over 2 laps, so stage 2 is needed.

## Stage 2: relocalisation by image retrieval (val, db = training run)

| method | pos RMSE | median | lat RMSE | yaw RMSE | ≤ 0.5 m | ≤ 1 m |
|---|---|---|---|---|---|---|
| 10 DINOv2-GeM retrieval, 7 cameras, top-10 | 1.70 m | 0.87 m | 0.75 m | 1.26° | 28 % | 56 % |
| 11 MegaLoc retrieval, 7 cameras, top-10 | 1.01 m | 0.42 m | 0.72 m | 1.09° | 58 % | 81 % |
| **12 MegaLoc + geometric re-ranking + 3-DoF multi-camera PnP** | **0.160 m** | **0.073 m** | **0.105 m** | **0.184°** | **98.6 %** | **100 %** |

Exp12 works per query in four steps:
1. MegaLoc similarities are averaged over the 7 cameras to retrieve candidate training frames.
2. For each candidate, the training frame is triangulated against a neighbouring frame using their training GT poses.
3. The query is matched with SuperPoint+LightGlue and a robust 3-DoF (x, y, yaw) PnP is solved over 4 cameras.
4. The first candidate that verifies is accepted. If none does, all 7 cameras are tried, and after that the
   retrieval pose is used.

A camera-to-GT clock offset of −9.1 ms was calibrated on the tune lap. All 2230 val queries were verified. Runtime is about
410 ms per image set (not optimised).

## Stages 3–4: real-time online localiser (final method)

Current protocol (lap cross-validation, `outputs/21_online/cv/`):

| Exp21 | pos RMSE | median | lat RMSE | yaw RMSE | ≤ 0.5 m | ≤ 1 m |
|---|---|---|---|---|---|---|
| F1 (val L1, L6) | 0.081 m | 0.060 m | 0.049 m | 0.063° | 100 % | 100 % |
| F2 (val L2, L7) | 0.238 m | 0.084 m | 0.211 m | 0.202° | 96.5 % | 98.5 % |
| F3 (val L3, L4) | 0.097 m | 0.061 m | 0.061 m | 0.085° | 100 % | 100 % |
| **F1–F3 pooled (6 laps)** | **0.156 m** | **0.066 m** | **0.131 m** | **0.132°** | **98.8 %** | **99.5 %** |
| F4 (val L0 out-lap, L4, L7; 5-lap map) | 0.464 m | 0.092 m | 0.404 m | 0.321° | 85.6 % | 94.3 % |
| L5 tuning lap under the 4 maps | 0.094 m | 0.062 m | 0.065 m | 0.091° | 99.9 % | 100 % |

Weak spots: L2 drives a different line from all other laps (up to 14 m off), and the pit exit of L0 is unmapped
in F4. Both are areas without nearby map passes.

Legacy split (val = 897–1120 s ≈ L6 + L7, earlier parameter set):

| method (legacy val, 2 laps) | pos RMSE | median | lat RMSE | yaw RMSE | ≤ 0.5 m | ≤ 1 m | runtime |
|---|---|---|---|---|---|---|---|
| 12 retrieval + PnP (stage 2) | 0.160 m | 0.073 m | 0.105 m | 0.184° | 98.6 % | 100 % | ~410 ms |
| 21 car-state EKF + cached map + guided matching + 6-DoF PnP, 7 cameras | 0.097 m | 0.059 m | 0.067 m | 0.094° | 100 % | 100 % | 40.7 ms |

1. **Offline map (Exp20).** Every training frame of every camera: SuperPoint keypoints matched to the previous and next
   frame, triangulated with the training GT. Stored per point: pixel, descriptor, 3D position (20 M points).
2. **Predict.** EKF on (x, y, yaw), driven by car-state vx, vy, wz (only samples up to the image stamp). No visual
   odometry: the car state is ~4× more accurate than our best VO over 1–10 s (stage 1), and with a fix every frame the
   prediction only bridges 0.1 s.
3. **Measure.** Map frames of the 3 nearest passes; their 3D points are projected with the predicted pose and matched
   to the query's SuperPoint keypoints in a small window (no LightGlue, no retrieval while tracking). A robust
   multi-camera PnP over (x, y, yaw) plus Δroll, Δpitch, Δz with priors, then a second, tighter matching pass.
4. **Update** with a Mahalanobis gate; a "challenger" replaces the state when 8 gated fixes agree with its first
   fix propagated by dead reckoning; MegaLoc retrieval + wide-radius PnP relocalises after 1 s without a fix where map
   exists, and only accepts a jump that is plausible for the elapsed time or has ≥ 300 inliers (recovers 3–80 m
   kidnaps in 1–2 s).

Key findings: one narrow camera is near-degenerate (0.41 m on the tune lap), side and rear cameras fix it; estimating
pitch/roll relative to the map frame improves lateral error by ~10 %; GT `pos_accuracy` spikes (2–3 m, same spots on
most laps) poison the map unless such frames are avoided; the final error has no bias and sits near the GT's own
accuracy (~0.1 m). Robust to gamma, gain, noise and blur on the query images (≤ 0.119 m on the dev laps); real
daylight against the night map is untested and remains the main risk. Track-border edge alignment was tried and
dropped (Exp22): with cameras 0.34–0.42 m above the asphalt, a 0.5 m lateral shift moves the borders by only ~1 px.

## Reproduce

```bash
source /scratch/local/velikanov/miniconda/bin/activate mast3r-slam    # GPU jobs inside tmux (see CLAUDE.md)
python scripts/extract_mcap.py data/<run>/<run>_0.mcap data/extracted
python scripts/exp01_deadreckoning.py
python -m odo.feature_vo --split tune --out outputs/05_planar_vo/steps_tune   # and --split val
python -m odo.planar_vo --matches outputs/05_planar_vo/steps_val --out outputs/05_planar_vo/theta_val
bash scripts/stage1_val.sh
python -m odo.vpr --model megaloc --cams fl fr r rl rr cl cr --splits train val --out outputs/10_vpr_desc
bash scripts/stage2_retrieval_val.sh && bash scripts/exp12_val.sh
python scripts/summary_report.py stage1 && python scripts/summary_report.py stage2
# stages 3-4 (GPU, in tmux)
bash scripts/exp20_map.sh                                   # map of the training run, ~10 min per camera
python scripts/sweep_online.py final '{}'                   # dev laps (tune + dev2)
python scripts/exp21_val.py                                 # legacy val
python -m odo.vpr --model megaloc --cams fl fr cl cr r rl rr --splits L8 --out outputs/10_vpr_desc
for f in F1 F2 F3 F4; do python scripts/exp21_cv.py $f --video; done; python scripts/cv_summary.py   # lap CV
python scripts/summary_report.py stage3
bash scripts/run_submission.sh <test_run.mcap> outputs/submission   # hidden test run -> submission.csv
```

## Sources consulted for method choice

* VGGT-Long (ICRA 26): https://arxiv.org/abs/2507.16443, code https://github.com/DengKaiCQ/VGGT-Long (Pi3/MapAnything backbones)
* VGGT-Motion: https://arxiv.org/html/2602.05508v1 (reports that MASt3R-SLAM/CUT3R hit OOM or lose tracking outdoors)
* VGGT-SLAM++: https://arxiv.org/html/2604.06830v1
* Keyframe-based feed-forward VO: https://arxiv.org/abs/2601.16020
* MegaLoc: https://arxiv.org/abs/2502.17237, code https://github.com/gmberton/MegaLoc
* LightGlue / SuperPoint: https://github.com/cvg/LightGlue
* Planar, non-holonomic 1-point motion model: Scaramuzza 2009 (IJCV / ICCV, "1-point RANSAC" for car-like motion)
* Pre-trained models used (declared): SuperPoint and LightGlue (cvg/LightGlue weights), MegaLoc (gmberton/MegaLoc).
  No external datasets.
