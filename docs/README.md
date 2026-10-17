# Camera-based localisation on a race track: prototype (stages 1 and 2)

Task: estimate the x, y and yaw of the race car (`constructor0/base_link` in `map`) from its 7 cameras.
Data: one night session at Yas Marina, 1194 s long. The score is position RMSE (see the dataset README and
`tools/evaluate.py`). Progress log: [`progress_log.md`](progress_log.md). Each experiment has its own README in `outputs/<exp>/`.

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
| `scripts/` | data extraction, experiment runners, parameter sweeps, summary reports |
| `outputs/NN_*` | one folder per experiment (README, PDF report, HTML, video, predictions) |
| `outputs/stage{1,2}_summary/` | method comparison: `summary.pdf`, `overlay.html`, `metrics.csv` |

## Protocol

* **Splits.** train covers t = 0 to 897 s (the map / database). The tune lap (t = 632 to 752 s) is inside train and is
  the only data used to choose parameters. val covers t = 897 to 1120 s: the last two full laps, which are also the fastest.
  Val numbers are computed once, with parameters frozen.
* **Anchoring.** Odometry starts from the GT pose of the first val frame. The rules allow the first pose of the run;
  for a segment that starts mid-run, this is the equivalent.
* **Allowed inputs.** Images, intrinsics and extrinsics, track borders. From the car state: v, vx, vy, wz, ax, ay.
  The training-run GT may serve as a map. Neither stage uses GT from the query time.
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
```

## Sources consulted for method choice

* VGGT-Long (ICRA 26): https://arxiv.org/abs/2507.16443, code https://github.com/DengKaiCQ/VGGT-Long (Pi3/MapAnything backbones)
* VGGT-Motion: https://arxiv.org/html/2602.05508v1 (reports that MASt3R-SLAM/CUT3R hit OOM or lose tracking outdoors)
* VGGT-SLAM++: https://arxiv.org/html/2604.06830v1
* Keyframe-based feed-forward VO: https://arxiv.org/abs/2601.16020
* MegaLoc: https://arxiv.org/abs/2502.17237, code https://github.com/gmberton/MegaLoc
* LightGlue / SuperPoint: https://github.com/cvg/LightGlue
* Planar, non-holonomic 1-point motion model: Scaramuzza 2009 (IJCV / ICCV, "1-point RANSAC" for car-like motion)
