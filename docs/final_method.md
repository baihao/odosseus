# Final method: camera localisation against a night map

One page on what runs on the test run (`bash scripts/run_submission.sh <test.mcap>`), why it is built this way, and
how well it works, including on daylight images. Code: `odo/online.py` (`DEFAULT` = the final parameters),
`odo/map_build.py`. History and all experiments: [`README.md`](README.md), [`experiments_explained.md`](experiments_explained.md).

## Inputs

* 7 synchronised cameras at 10 Hz, intrinsics and extrinsics.
* Car state: vx, vy, wz, v (allowed fields only).
* The first pose of the run.
* The training run (night) as a prebuilt map. Nothing from the test run is ever added to the map.

## Offline: the map (once, from the training run)

For every training frame of every camera: SuperPoint keypoints, matched with LightGlue to the previous and next frame,
triangulated with the training GT poses. Frames where the GT's own accuracy flag spikes are skipped. Result: 20 M 3D
points with descriptors, plus MegaLoc global descriptors of every training image set (for relocalisation).

## Online, per 7-camera image set (causal, ~41 ms on an H100)

1. **Predict.** EKF on (x, y, yaw), driven by the car-state velocities and yaw rate. No visual odometry: over 1–10 s
   the car state is ~4× more accurate than our best VO.
2. **Match.** Map frames from the 3 nearest passes are projected with the predicted pose. Each map point is matched to
   a query keypoint within a small window (mutual best descriptor). Everything runs on the GPU; there is no LightGlue
   or retrieval while tracking.
3. **Solve.** A robust multi-camera PnP over x, y, yaw, plus small roll/pitch/height corrections, followed by a second,
   tighter matching pass.
4. **Update**, with a Mahalanobis gate:
   * **Weak fixes (< 60 inliers) update position only.** Their yaw is ignored, and heading comes from the car-state
     yaw rate (Exp24).
   * **Challenger.** Rejected fixes that agree with each other for 8 frames take over (escapes a wrong state).
   * **Relocalisation.** After 1 s without a fix: MegaLoc retrieval + wide-radius PnP. A jump is accepted only if it is
     plausible for the elapsed time (position 0.5 m + 2 m/s; **yaw 0.3° + 0.1°/s**, Exp24) or has ≥ 300 inliers.
5. **Output** a pose for every image set (100 % coverage).

## Accuracy

| condition | result |
|---|---|
| night, lap cross-validation F1–F3 (6 laps, Exp21) | 0.156 m RMSE, 0.131 m lateral, 0.13° yaw; laps 0.07–0.12 m except the off-line lap L2 (0.30 m) |
| night, tuning lap L5 (final parameters) | 0.089 m |
| **day proxy** of the hardest lap (F5/L2; night map) | 0.200 m (day), 0.190 m (day, lights removed) vs 0.299 m night |
| **day proxy** (lights removed) of an untouched lap (F5/L7) | 0.082 m vs 0.089 m night |
| day proxy + only 10 % of keypoints / + overexposure | 0.39 m / 0.45 m |
| kidnaps of 3–80 m, up to 20° | recovered in 1.0–3.4 s |

## Why we expect it to transfer to daylight (Exp24, `outputs/24_day/summary.md`)

* On night→day translated images (CycleGAN-Turbo, also with all light sources removed), guided matching keeps
  65–75 % of the night inliers and a map fix in ~97 % of frames. Accuracy is unchanged.
* MegaLoc relocalisation finds a candidate within 15 m for 100 % of day queries.
* The system tolerates much worse matching than the proxy shows (4× fewer features: no loss). Two rules keep a few
  weak or wrong fixes from corrupting the heading, which was the failure mode under heavy degradation.
* If matching fails entirely, the filter falls back to car-state dead reckoning and relocalises as soon as matching
  returns.

Residual risk: real sun shadows, glare and lens flare are not in the proxy. The stress tests bound how much worse
matching can get before accuracy suffers (below ~10 % of the features).
