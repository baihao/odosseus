# Final method: camera localisation against a night map (Exp24)

What runs on the test run (`bash scripts/run_submission.sh <test.mcap>`), what Exp24 changed, and the experiments
behind it. Code: `odo/online.py` (`DEFAULT` = the final parameters), `odo/map_build.py`. Full history:
[`README.md`](README.md), [`experiments_explained.md`](experiments_explained.md). Exp24 run table and reports:
[`outputs/24_day/summary.md`](../outputs/24_day/summary.md).

## 1. The method

**Inputs.** 7 synchronised cameras at 10 Hz with intrinsics and extrinsics; car state vx, vy, wz, v; the first pose
of the run; the training run (night) as a prebuilt map. Nothing from the test run is ever added to the map.

**Offline map (once, from the training run).** For every training frame of every camera: SuperPoint keypoints, matched
with LightGlue to the previous and next frame, triangulated with the training GT poses. Frames where the GT's own
accuracy flag spikes are skipped. Result: 20 M 3D points with descriptors, plus MegaLoc global descriptors of every
training image set (for relocalisation). The test-run map uses all 9 night laps.

**Online, per image set (causal, ~41 ms with 7 cameras on an H100).**
1. **Predict.** EKF on (x, y, yaw), driven by the car-state velocities and yaw rate. No visual odometry: over 1–10 s
   the car state is ~4× more accurate than our best VO.
2. **Match.** Map frames from the 3 nearest passes are projected with the predicted pose. Each map point is matched to
   a query keypoint inside a small window (mutual best descriptor), all on the GPU.
3. **Solve.** Robust multi-camera PnP over x, y, yaw plus small roll / pitch / height corrections, then a second,
   tighter matching pass. A fix needs at least 30 inliers.
4. **Update** through a Mahalanobis gate.
   * **Weak fixes (< 60 inliers) update position only**; heading comes from the car-state yaw rate. *(Exp24)*
   * **Challenger:** rejected fixes that agree with each other for 8 frames take over (escape from a wrong state).
   * **Relocalisation:** after 1 s without a fix, MegaLoc retrieval + wide-radius PnP. A jump is accepted only if it is
     plausible for the elapsed time (position 0.5 m + 2 m/s, **yaw 0.3° + 0.1°/s** *(Exp24)*) or has ≥ 300 inliers.
5. **Output** a pose for every image set (100 % coverage).

## 2. What changed in Exp24

The test run is in daylight, the map is night-only, and we may not map the test run. Exp24 tested the Exp21 system on
synthetic day images and found that day images alone are no problem, but *weak* matching (few or wrong inliers)
could corrupt the heading. Two rules fix it; both act only when matching is weak, so well-matched laps are unchanged.

| rule | Exp21 | final | failure it fixes |
|---|---|---|---|
| relocalisation yaw plausibility | 1° + 2°/s | **0.3° + 0.1°/s** (`reloc_yaw0_deg`, `reloc_yaw_rate_deg`) | an overexposed run accepted a relocalisation 2.5° off in yaw, then drifted 7 m sideways |
| yaw from weak fixes | always used | **ignored below 60 inliers** (`yaw_min_inl`) | a 30-inlier fix turned yaw by 1.6°; the filter then rejected the correct fixes: 11 m error for 25 s |

The yaw bound is still ~3× the real dead-reckoning drift (~0.09° in 1 s, ~0.4° in 10 s); larger jumps still pass with
≥ 300 inliers or 8 consistent fixes. With a full covariance kept for relocalisation / reset, kidnaps of 3–80 m and up
to 20° still recover in 1.0–3.4 s. Nothing else (map, matching, PnP, noise levels) changed.

## 3. Exp24 experiments

**Setup.** No day data exists, so day *proxies* of the query images were made and matched against the unchanged night map:
* **day:** CycleGAN-Turbo `night_to_day` (`scripts/day_translate.py`); geometry kept (1.5 px), bright sky, day colours.
* **day_nolights:** the same, with all light sources (pixels ≥ 200) removed first, since lamps are off by day.
* **stress** on day_nolights: 50 / 25 / 10 % of the keypoints kept, overexposure (gain 1.6), gamma 1.8.

Main lap: F5/L2 (map = night L0 L1 L3 L4 L6 L8). L2 is the hardest lap, driven up to 14 m off the mapped lines.
Checks on L5 (F1, the tuning lap) and on the untouched lap F5/L7. No full cross-validation was run.

**Accuracy (position RMSE).**

| query images | Exp21 | final |
|---|---|---|
| F5/L2 night | 0.302 m | 0.299 m |
| F5/L2 day | 0.200 m | – (yaw gate only: 0.200 m) |
| F5/L2 day, lamps removed | 0.220 m | **0.189 m** |
| F5/L2 day, lamps removed, 10 % keypoints | 2.457 m | **0.390 m** |
| F5/L2 day, lamps removed, gain 1.6 | 1.243 m | **0.450 m** |
| F1/L5 night (tuning lap) | 0.089 m | 0.089 m |
| F5/L7 night / day, lamps removed (untouched lap) | 0.089 / 0.082 m | 0.089 / 0.082 m |
| F7/L2 day / day, lamps removed (all night laps in map) | – | 0.066 / 0.067 m |

* Day queries keep 65–75 % of the night inliers (F7: ~60 %) and get a map fix in ~97 % of frames (F7: 100 %).
* MegaLoc relocalisation transfers: the best of 4 candidates is within 15 m for 100 % of day queries (night 99.4 %).
* F7 mirrors the test-run setting but is an upper bound: L2's own night frames are in the map, so only appearance
  changes, not the viewpoint. F5/L2 (off-line lap) and F5/L7 (on-line lap) are the realistic estimates.

**Two cameras only (cl + cr), F5/L2, final configuration.** Night 0.287 m, day (lamps removed) 0.265 m, F7 night
0.067 m. Accuracy matches 7 cameras when matching is good, but with 10 % of the keypoints the day error is 0.80 m
(7 cameras: 0.39 m).

**Lower inlier minimum for a fix (30 → 15), cl + cr, F5/L2.** Night unchanged (0.287 m), day 0.265 → 0.255 m. With 10 %
of the keypoints it doubles the fixes but they are noisy (night 0.46 → 0.51 m, day 0.80 → 0.81 m). No gain, kept at 30.

**Tried and dropped.**
* **No keypoints on light sources:** night L2 4.2 m. Lamps carry along-track information at night; by day they simply
  don't match, so removing them gains nothing.
* **Weak-fix threshold 100 instead of 60:** night L2 0.61 m.
* **A map built from the test run** (day-to-day matching): not allowed, removed.
* **Other day generators:** Qwen-Image-Edit gives realistic sun and shadows but shifts the image (13.6 px, 2.9 px after
  homography registration) and changes the appearance a lot; CAP-VSTNet style transfer is pixel-aligned (1.5 px) but
  looks like dusk, not day.

**Limits.** The proxies have no real sun shadows, glare or lens flare. The stress tests give the margin: accuracy holds
down to ~25 % of the features and through overexposure, and degrades gracefully (0.39 m) at 10 %. If matching fails
entirely, the filter falls back to car-state dead reckoning (~10 m drift per lap) and relocalises when matching returns.
