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
     plausible for the elapsed time (**position 0.5 m + 0.3 m/s, yaw 0.3° + 0.1°/s** *(Exp24)*) or has ≥ 300 inliers.
5. **Output** a pose for every image set (100 % coverage).

## 2. What changed in Exp24

The test run is in daylight, the map is night-only, and we may not map the test run. Exp24 tested the Exp21 system on
synthetic day images and found that day images as such cost little accuracy, but *weak* matching (few or wrong inliers)
let single bad measurements take over the filter. Three rules fix it. All act only when matching is weak or a
relocalisation jump is large, so well-matched laps are unchanged.

| rule | Exp21 | final | failure it fixes |
|---|---|---|---|
| relocalisation yaw plausibility | 1° + 2°/s | **0.3° + 0.1°/s** (`reloc_yaw0_deg`, `reloc_yaw_rate_deg`) | an overexposed run accepted a relocalisation 2.5° off in yaw, then drifted 7 m sideways |
| relocalisation position plausibility | 0.5 m + 2 m/s | **0.5 m + 0.3 m/s** (`reloc_sig_rate`) | on Qwen day images, a relocalisation with 80 inliers (all from one camera) jumped 21 m along the track after 5.7 s without a fix; the filter stayed 20 m off to the end of the lap |
| yaw from weak fixes | always used | **ignored below 60 inliers** (`yaw_min_inl`) | a 30-inlier fix turned yaw by 1.6°; the filter then rejected the correct fixes: 11 m error for 25 s |

The yaw bound keeps ~3–4× over the measured dead-reckoning yaw drift (σ 0.07° in 1 s, 0.29° in 10 s). The position
bound is about the measured p95 drift itself: dead reckoning drifts ~0.3 m/s at p95 over 1–10 s
([`parameters.md`](parameters.md) 2.1; an earlier estimate of 0.06 m/s averaged over whole laps, where errors partly
cancel). Larger jumps still pass with ≥ 300 inliers or 8 consistent fixes. Which parameters are tuned and how most of
them can be measured, derived or estimated online: [`parameters.md`](parameters.md). Kidnaps of 3–80 m
and up to 20° still recover in 1.0–3.4 s. Nothing else (map, matching, PnP, noise levels) changed.

## 3. Exp24 experiments

**Setup.** No day data exists, so day *proxies* of the query images were made and matched against the unchanged night map:
* **day:** CycleGAN-Turbo `night_to_day` (`scripts/day_translate.py`); geometry kept (1.5 px), bright sky, day colours.
* **day_nolights:** the same, with all light sources (pixels ≥ 200) removed first, since lamps are off by day.
* **day_qwen:** Qwen-Image-Edit-2509 + Lightning ("midday sun, all lights off"), cameras cl and cr of L2 only. The
  hardest proxy: real-looking sun and shadows, but it also redraws textures and small details, so it changes the
  appearance more than the other proxies. Registered onto the night frames with one homography each (2.9 px).
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

**Two cameras only (cl + cr), L2.** The 7-camera runs above were measured before the position rule. Re-run with it,
L5, 10 % keypoints and gain 1.6 give the same numbers.

| query images, cl + cr | Exp21 | yaw rules only | final (+ position rule) |
|---|---|---|---|
| F5/L2 night | 0.284 m | 0.287 m | 0.287 m |
| F5/L2 day, lamps removed | – | 0.265 m | 0.265 m |
| F5/L2 day_qwen | 5.20 m | 5.20 m | **0.419 m** (p95 0.91 m, 96 % ≤ 1 m) |
| F7/L2 night / day_qwen | – | 0.067 / 0.157 m | 0.067 / 0.157 m |

* On Qwen images only 43 % of frames get a map fix (night 94 %). Dead reckoning bridges the gaps, and the error
  stays under 1 m for 96 % of the lap. One wrong relocalisation caused the whole 5.2 m; the position rule rejects it.
* With 10 % of the keypoints, 2 cameras are much more fragile than 7 (day, lamps removed: 0.80 m vs 0.39 m).

**Lower inlier minimum for a fix (30 → 15), cl + cr, F5/L2.** Night unchanged (0.287 m), day 0.265 → 0.255 m. With 10 %
of the keypoints it doubles the fixes but they are noisy (night 0.46 → 0.51 m, day 0.80 → 0.81 m). No gain, kept at 30.

**Tried and dropped.**
* **No keypoints on light sources:** night L2 4.2 m. Lamps carry along-track information at night; by day they simply
  don't match, so removing them gains nothing.
* **Weak-fix threshold 100 instead of 60:** night L2 0.61 m.
* **Looser descriptor threshold when matches are few** (`sim1_lo=0.65`, cl + cr): Qwen 5.20 → 0.32 m (65 % fixes), but
  night 0.287 → 0.675 m (wrong matches pull the pose sideways).
* **A map built from the test run** (day-to-day matching): not allowed, removed.
* **CAP-VSTNet style transfer** as a day proxy: pixel-aligned (1.5 px) but it looks like dusk, not day.

**Limits.** Only the Qwen proxy has sun and shadows, and only for 2 cameras; none has glare or lens flare. The stress
tests give the margin: accuracy holds down to ~25 % of the features and through overexposure, and degrades gracefully
(0.39 m) at 10 %. On Qwen images, 2 cameras give 0.42 m with half the frames unmatched. If matching fails
entirely, the filter falls back to car-state dead reckoning (~10 m drift per lap) and relocalises when matching returns.
