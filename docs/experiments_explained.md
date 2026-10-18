# Experiments explained: what was tried, what worked, what failed, and why

This document is for a computer vision engineer who is not a localisation or tracking specialist. It explains
the reasoning behind stages 1 (visual odometry), 2 (relocalisation) and 3–4 (the final real-time system, section 6),
along with the numbers. Short summaries are in
[`README.md`](README.md); the chronology is in [`progress_log.md`](progress_log.md). Each experiment has its own folder in
`outputs/NN_*/` with a README, a PDF report, an interactive map (`traj_*.html`) and a debug video.

The questions for the dataset provider are at the end, in [section 7](#7-questions-for-the-dataset-provider).

---

## 1. The problem in tracking terms

We have to output, at 10 Hz and **causally** (using only images up to time t), the planar pose
(x, y, yaw) of the car's `base_link` in a fixed `map` frame. The score is the position RMSE against the ground truth.
The ground truth comes from the car's own state estimator (RTK GNSS + lidar + IMU), with a stated accuracy of about
0.1 m / 0.2°.

There are two kinds of information available, and almost every localisation system combines them:

| kind | what it measures | strength | weakness |
|---|---|---|---|
| **Relative motion (odometry)**: wheel/IMU signals, visual odometry (VO) | how the car moved between t−1 and t | smooth, always available, very accurate over short times | errors **accumulate** (drift): a 0.1° heading error turns into metres of lateral error after a few hundred metres |
| **Absolute fixes**: matching the current view to a map | where the car is in the map frame | no drift | can fail completely (wrong place, too few matches), noisier frame to frame |

The standard design is a **filter** (Kalman filter or particle filter). It *predicts* the pose with odometry and
*corrects* it whenever a trustworthy absolute fix arrives. We built and measured the two ingredients separately
(stage 1 = odometry, stage 2 = absolute fixes). The filter that joins them is stage 4 (section 6), the final system.

### Metrics used

* **Position RMSE / median / p95**: the official score, computed in absolute map coordinates. `odo/metrics.py`
  reproduces `tools/evaluate.py` exactly (checked on two files).
* **Lateral / longitudinal error**: the error expressed in the GT car frame. Lateral (across the track) matters most to the
  car's controller. Longitudinal (along the track) is typically the harder one on long straights.
* **RPE (relative pose error) over 1 s and 10 s**: how much the *relative* motion over a 1 s or 10 s window differs from
  GT. This measures odometry quality without the accumulated drift, so it is the right metric for comparing VO methods.
  "1.25 % / 1.26°" means 1.25 % of the distance travelled in 10 s as translation error, plus 1.26° of heading error.
* For a drifting method the absolute RMSE depends mainly on how long it has run. A pure-odometry result (stage 1)
  should therefore be judged by RPE, while an absolute method (stage 2) is judged by RMSE.

### What "tuning" means here

**Tuning = choosing the settings of a method by measuring its error on data with known GT, then keeping the
setting that worked best.** Nothing is trained in this project in the deep-learning sense: all networks (SuperPoint,
LightGlue, MegaLoc, DINOv2) are used with their published weights. What we tune is a few dozen numbers that the
methods need, such as thresholds, noise levels, how many map frames to use, or a clock offset.

Every time a setting is chosen because it gave a lower error on some lap, that lap has influenced the method. Its
error is then no longer an honest estimate of how the method will do on new data. That is why one lap is
reserved for tuning and never used to report the final numbers. Under the current protocol it is **L5**, which is
also never in a map. The validation laps are run only after all settings are frozen. Under the legacy split the tuning
data were the tune lap (632–752 s ≈ L4) and, for stage 4, the dev2 lap (752–897 s ≈ L5).

How tuning works in practice: run the method on the tuning lap with several values of a setting (a "sweep"),
compare position / lateral / yaw RMSE, keep the best, freeze it. When the differences are within noise, keep the
simpler or faster option.

What was tuned, per stage:

| stage | tuned on the tuning lap | example |
|---|---|---|
| 1, VO | per-camera rotation gain, bias and noise σ; fusion scheme | `fr` reports 93.5 % of the true rotation → gain correction |
| 2, retrieval + PnP | number of retrieved candidates, averaging over cameras, jump guard, and the **image-vs-GT clock offset** | −9.1 ms, fitted from longitudinal error vs. speed |
| 4, online system (`odo/online.py: DEFAULT`) | cameras used (5 vs 7), 3-DoF vs 6-DoF PnP, measurement noise floor (0.10 m / 0.23°), number of map passes (3), matching thresholds, keypoints (2048), relocalisation gate, challenger rule (8 frames, fixed seed), map selection radius (15 m) | the challenger and relocalisation rules were re-tuned on L5 under all fold maps after the CV run exposed two failures |

Settings that were **not** tuned but set from physics or the data: the camera calibration and car-body crop, the
car-state signals used, GT-accuracy threshold for map frames (1 m), PnP priors (0.5° roll/pitch, 5 cm height),
process noise of the filter. These came from reasoning and were checked on the tuning lap, not optimised.

The stage-4 sweep turned out flat: matching thresholds, map passes, noise floor and keypoint count all gave
0.103–0.108 m on the development laps. So the result does not hinge on a lucky parameter choice. The two changes
that mattered (the recovery rules) fixed failure modes rather than shaving millimetres.

---

## 2. Data observations that shaped the methods

* **Speed.** Up to 40 m/s at 10 Hz, so the car moves **3–4 m between consecutive frames**. Most SLAM/VO systems
  assume tens of centimetres of motion per frame. This is the main reason the off-the-shelf systems failed (section 3).
* **Night.** The track is flood-lit, the sky is black, and the asphalt has little texture. Most stable features are
  lights, grandstands, barriers and kerbs.
* **Car body in view.** `fr`, `cl`, `cr` and `rr` see parts of the car (nose, wheel, side pods), so the bottom of the
  image is cropped (`odo/prep.py`). Features on the car body move with the camera and would bias any motion estimate.
* **Camera timing.** All 7 cameras share the same `header.stamp` to within 0.025 ms (hardware trigger). One correction:
  `outputs/00_data_overview/README.md` says the stamps are "not synchronised"; that note is out of date. The `r` camera
  drops 5 frames over the run. The images were logged about 41 ms after their stamp (`log_time − stamp`).
* **Calibration quality** (dataset README): extrinsic pitch is reliable for all cameras. Yaw is reliable only for
  `r`/`rl`, ±1° for `fl`/`fr`, and `cl`/`cr` use an older calibration. Roll and translation are nominal mount values.
  Our VO measured this directly: each camera under-estimates rotation by a different factor (`fr` 0.935, `cr` 0.876).
  This points to intrinsic (focal length / distortion) errors as well.
* **Suspicious intrinsics.** `fr`, `cl`, `cr` and `rr` have fx/fy ≈ 1.15 (for example `cr`: fx = 744, fy = 644), while
  `fl`, `r` and `rl` have fx ≈ fy. Pixels are almost certainly square, so either the images were resized
  anisotropically or the K matrices belong to another resolution (see questions).
* **Car state.** The permitted signals (speed, vx, vy, yaw rate wz, ax, ay) are excellent. Integrating them alone
  ("dead reckoning") gives a yaw-rate noise of about 0.01° per frame. `valid_imu_b` is `False` for the entire run,
  but wz still looks like a gyro signal (see questions).

---

## 3. Stage 1: visual odometry

**Goal.** Estimate the car's motion between consecutive frames from images, then integrate it from the first pose.
Speed from the car state is allowed and provides metric scale. Monocular VO cannot recover scale by itself.

**Reference to beat.** Exp01, dead reckoning from the car state (no images at all): **20.5 m** position RMSE over the
two val laps, RPE 1 s yaw 0.087°, RPE 10 s 0.57 % / 0.36°. Its drift is slow: about 0.02°/s of gyro bias and ~0.5 % speed
scale error.

### 3.1 Failures: learned / foundation-model SLAM

| exp | method | result | why it failed |
|---|---|---|---|
| 02 | **MASt3R-SLAM** (learned dense matching + SLAM back-end) | Tracking lost after 1–3 frames on every camera tried, with calibrated and uncalibrated intrinsics and with relaxed thresholds. Never relocalised. | It matches each frame against a keyframe's dense point map. With 3–4 m between frames, a dark scene and distant lights, too few matches are geometrically consistent, so it declares tracking lost. Published reports describe the same behaviour on fast outdoor sequences. |
| 03 | **VGGT-Long / Pi-Long** (feed-forward multi-view reconstruction in 60-frame chunks, aligned with Sim(3)) | Never loses track, but heading accuracy is poor: RPE 1 s yaw 1.6°, 313 m RMSE on val (`fr` camera). VGGT backbone 48 m vs Pi3 12 m on a 40 s clip, where dead reckoning gets 3.6 m. | These networks are trained mostly on slow, densely sampled, well-lit scenes. Their per-frame rotation is 5–20× noisier than two-view geometry here, and the translation direction often flips sign. Also, as run here they see 60 future frames, so they are not causal. They were evaluated only as an upper bound. |

**Lesson.** At high speed, low frame rate and night lighting, classical calibrated two-view geometry with a strong
learned *matcher* beats end-to-end learned SLAM.

### 3.2 Partial success: calibrated feature VO (Exp04)

For each camera and each pair of consecutive frames:

1. SuperPoint keypoints + LightGlue matching (learned features, but classical geometry afterwards).
2. Undistort the keypoints and estimate the **essential matrix** with MAGSAC++. This gives the camera's rotation and
   translation *direction*.
3. Express the rotation in the car frame through the extrinsics and keep the yaw increment.

The 7 per-camera yaw increments are then **fused**:
* Per-camera gain, bias and noise σ are fitted on the tune lap. This absorbs the calibration errors described above,
  such as a camera that reports 93.5 % of the true rotation.
* Three fusion schemes were tried:
  * **Median + temporal gate.** It locked up in turns, because the gate rejected genuine changes in yaw rate.
  * **Constant-rate Kalman filter.** It lagged at hard turn-ins (−8° within 1 s).
  * **Per-step robust average (IRLS with a Cauchy loss).** No motion model and no lag. This one was kept.
* Translation: the VO translation direction was 3–11° off, so the step length comes from car speed along the heading.

Val: **50.9 m RMSE**, RPE 1 s yaw 0.79°. The median per-step yaw error is excellent (0.04°), but 8–22 % of steps are gross
outliers, and the side cameras have a bias (`rr` −0.5° per step).

### 3.3 Best camera-only VO: planar, Ackermann-constrained, multi-camera (Exp05)

The key idea is to **remove degrees of freedom that we already know**. A car on a track moves almost in a plane and
(almost) does not slide sideways, so between two frames its motion is one rotation θ about the vertical axis plus a
chord of a circular arc. The chord length comes from the car speed. The extrinsics are known, so every camera's relative pose is
a function of θ alone, including the lever-arm effect of cameras that sit far from the rear axle. For each camera, θ is
found by a robust 1-D search over the matched points (truncated Sampson error). This is the "1-point" non-holonomic
model of Scaramuzza (2009), extended to 7 cameras.

Planar θ and essential-matrix yaw are fused as 14 measurements per step (7 cameras × 2 estimators), with the same
tune-lap calibration and IRLS. Tune lap: 8.2 m RMSE, vs 11.4 m for essential-only and 7.2 m for dead reckoning.

Val: **43.3 m RMSE, yaw RMSE 3.1°, RPE 1 s yaw 0.34°, RPE 10 s 1.25 % / 1.26°**. This is the best camera-only VO, but it
is still ~2× worse than dead reckoning.

### 3.4 Can VO help the car-state odometry? (Exp06)

The VO was used only to estimate the gyro's slow bias (gyro − VO, low-pass filtered). For every filter time constant
on the tune lap, the result was *worse* than plain dead reckoning; val 20.6 m vs 20.5 m.

**Stage-1 conclusion.**
* The car-state yaw rate is about 4× more accurate than our best VO over 1–10 s. VO's own slow error (~1° per 10 s)
  is larger than the gyro bias (~0.02°/s), so VO cannot correct it.
* Odometry of any kind drifts to ~20 m over two laps. **The useful role of cameras is absolute localisation, not
  odometry.**
* Remaining VO error sources, in our judgement: pitch under braking and kerb strikes (breaks the planar assumption),
  uncalibrated side-camera extrinsics, and motion blur on the side cameras.

---

## 4. Stage 2: absolute localisation against a map of the training run

The training run has GT poses for every image, so it can serve as a geo-referenced map. The task slides
explicitly suggest this (method 3).

### 4.1 Coarse: place recognition / image retrieval (Exp10, Exp11)

Each image is turned into a global descriptor: one vector per image, trained so that the same place gives similar
vectors. The query image set is compared with all 8970 training image sets.

* The cosine similarity of each camera is **averaged over the 7 cameras**. This is valid because the cameras are
  synchronised, and it was the strongest single improvement in this part.
* Pose = similarity-weighted mean of the top-10 database poses that lie within 10 m of the best match.

| exp | descriptor | val RMSE | median | ≤ 1 m |
|---|---|---|---|---|
| 10 | DINOv2 (generic foundation features, CLS + GeM pooling) | 1.70 m | 0.87 m | 56 % |
| 11 | MegaLoc (DINOv2 fine-tuned for place recognition) | 1.01 m | 0.42 m | 81 % |

The correct place is retrieved almost always (R@1 within 10 m: 100 %). The remaining metre comes from the training car having
driven a slightly different line (lateral error) and from the database frames being 3–4 m apart (longitudinal error),
so the best match can be up to 2 m away. Retrieval alone cannot do better than the spacing of the map.

### 4.2 Fine: metric pose from 2D–3D matches (Exp12, the main result)

For each query image set:

1. **Retrieve** the top-50 training frames (MegaLoc, 7 cameras). Keep up to 3 candidates that are at least 8 m apart,
   to cover the case where the first match is the wrong place.
2. **Build a local 3D map** around the candidate. Take training frame A and a neighbouring frame B at least 1.5 m away,
   match them with SuperPoint+LightGlue, and **triangulate** the matches with their GT poses. This gives 3D points in the
   map frame. No prebuilt global point cloud is needed.
3. **Match query ↔ A** and solve **PnP**, restricted to 3 DoF (x, y, yaw). Height, roll and pitch are taken from
   frame A, since it is the same spot on the track. One car pose is solved jointly across 4 cameras (`fr`, `cl`, `cr`,
   `rr`) using the known extrinsics, with robust least squares (Huber, then inlier refit).
4. **Verify**: accept the pose if it has ≥ 30 inliers and jumps < 15 m from the retrieval pose. Otherwise try the next
   candidate, then retry with all 7 cameras, and fall back to the retrieval pose as a last resort.

**Clock offset.** On the tune lap, the longitudinal error was proportional to (query speed − map speed), which is the
signature of a timing offset. The fit gives **−9.1 ms** between the image stamps and the GT stamps (at 40 m/s that is
0.36 m). After correcting it, the longitudinal bias disappeared. We do not know its physical origin; see the questions.

Iterations on the tune lap:

| iteration | RMSE | median | what we learned |
|---|---|---|---|
| retrieval only | 1.77 m | 0.37 m | |
| PnP, first 300 queries | 0.168 m | 0.141 m | works; −0.14 m longitudinal bias → timing |
| PnP on the full lap | 1.15 m | 0.098 m | 4.3 % of queries fell back to a bad retrieval (jump guard too tight) |
| + 3-candidate re-ranking + −9.1 ms | 0.73 m | 0.071 m | 20 failures in one 2 s stretch |
| + 7-camera retry (final) | 0.48 m | 0.071 m | verified poses 0.165 m; 9 fallbacks with ~5 m error each dominate the RMSE |

**Val result, parameters frozen: 0.160 m RMSE, median 0.073 m, lateral 0.105 m, yaw 0.184°, 100 % within 1 m.**
All 2230 queries were verified (2229 on the first candidate). It runs per frame with no odometry or filter, and the
official `evaluate.py` gives the same numbers. Runtime is about 410 ms per image set; this is unoptimised and could be cut
roughly by 3× by caching the map features.

### 4.3 How much to trust 0.16 m

* **It is close to the GT's own accuracy.** The README states ~0.1 m. The state estimator's own `pos_accuracy` has a median
  of 0.30 m on val, and is > 0.5 m for 4 % of samples. Below roughly 0.1–0.2 m we may be measuring GT noise rather
  than our own error.
* **Map and val come from the same session.** Same night, lighting and calibration, recorded minutes apart. The challenge
  description says the hidden test run "may include higher speeds and different lighting conditions" and lists
  "sun, shadows, artificial light" among the difficulties. A daylight test run against a night map is the biggest
  risk for this method. Retrieval (MegaLoc) and SuperPoint matching are both appearance-based.
* **Runtime is scored.** It is 5 % of the score, with full points at or below the camera frame period (100 ms). At
  ~410 ms per image set, Exp12 is about 4× too slow as it stands.
* **The tail matters.** On the tune lap, 0.8 % of queries fell back to raw retrieval with ~5 m error, and these
  doubled the RMSE (0.165 → 0.48 m). On val there were none, partly by luck. The filter in stage 4 is
  meant to reject such outliers using odometry.
* **The map pose is GT-derived.** Triangulating with training GT means any GT error in the map transfers directly to the
  query. This is allowed ("the training run becomes a map"), but it assumes the test run uses the same `map` frame.

---

## 5. Summary of successes and failures

| | outcome | key reason |
|---|---|---|
| Dead reckoning (car state) | 20.5 m over 2 laps; best odometry | the IMU yaw rate is very good; drift is slow but unbounded |
| MASt3R-SLAM | ✗ fails after 1–3 frames | 3–4 m inter-frame motion, night, low overlap |
| VGGT-Long / Pi-Long | ✗ 313 m | rotation 5–20× noisier than two-view geometry; non-causal chunks |
| SuperPoint+LightGlue essential VO | ~ 50.9 m | good median, heavy-tailed outliers, calibration biases |
| Planar Ackermann multi-camera VO | ~ 43.3 m, best camera-only VO | prior knowledge (planar motion, known speed) removes DoF |
| VO-assisted gyro bias | ✗ no gain | VO drift > gyro drift |
| Retrieval, DINOv2 / MegaLoc | ✓ 1.70 / 1.01 m, no drift | 7-camera similarity averaging; limited by map spacing |
| Retrieval + triangulation + 3-DoF multi-camera PnP | ✓ 0.160 m, 410 ms | metric geometry on top of retrieval; planar DoF reduction; clock-offset fix |
| Track-border edge alignment | ✗ makes the pose worse | cameras 0.34–0.42 m high: 0.5 m lateral ≈ 1 px; wrong edges |
| **Car-state EKF + cached map + guided matching + 6-DoF PnP, 7 cams** | **✓ legacy val 0.097 m; lap CV 0.156 m (0.07–0.12 m on well-mapped laps), 41 ms** | prediction makes matching cheap and safe; all cameras; pitch/roll estimated; GT-quality-aware map; recovery checked against dead reckoning |

## 6. Stages 3–4: the final real-time system (Exp20–22)

### 6.0 Decisions at a glance

✓ = success kept, ✗ = failure (and its fix), – = tried and dropped. Numbers are position RMSE unless noted.

| # | decision / finding | outcome | what we did |
|---|---|---|---|
| 1 | Replace per-frame retrieval (Exp12, 410 ms) by tracking: car-state prediction + map | ✓ 0.16 → ~0.10 m, 410 → 41 ms | EKF predicted by vx, vy, wz; fixes from a cached map |
| 2 | Motion model: car state, **not** visual odometry | ✓ | car-state dead reckoning is ~4× more accurate than our best VO over 1–10 s (section 3); VO is not used in the final system |
| 3 | Cache the map once (SuperPoint + triangulation with training GT, 20 M points on the GPU) | ✓ | no LightGlue or retrieval while tracking |
| 4 | Guided matching: project map points with the predicted pose, search ±24 px | ✓ ~1 ms, more matches than LightGlue | |
| 5 | ✗ One narrow camera (`fr`): yaw and sideways shift nearly interchangeable, 0.41–0.53 m on a lap | fixed | use all 7 cameras in one PnP → 0.12 m |
| 6 | ✗ Training GT has 2–3 m `pos_accuracy` spikes at fixed spots; those frames poison the map | fixed | prefer map frames away from spikes (GT metadata of the training run) |
| 7 | Estimate Δpitch, Δroll, Δz relative to the map frame (6-DoF PnP with priors, planar output) | ✓ lateral −10 % | |
| 8 | ✗ First version 150 ms: GPU float64 Gauss-Newton was launch-bound, many GPU→CPU syncs | fixed | analytic Jacobian on the CPU, batched sync-free matching, GPU JPEG decoding → 40 ms |
| 9 | ✗ Kidnapped filter (10–80 m off) never recovered | fixed | MegaLoc relocalisation after 1 s without a fix, verified by wide-window PnP → recovery in 1–2 s |
| 10 | Track-border edge alignment | – dropped | cameras 0.34–0.42 m above the asphalt: 0.5 m lateral ≈ 1 px, below edge noise |
| 11 | Parameter sweep (matching thresholds, map passes, noise, keypoints) | ✓ but flat: 0.103–0.108 m | the floor is the GT and map consistency, not the matcher |
| 12 | Photometric stress (gamma, gain, noise, blur) | ✓ ≤ 0.119 m | |
| 13 | Switch from one val split to lap cross-validation (folds F1–F4, L5 = tuning) | ✓ more honest numbers | `docs/laps.md`, `docs/cv_performance.md` |
| 14 | ✗ CV: a relocalisation fired while the car was merely off the mapped lines and jumped 18 m to a wrong place (L2: 3.7 m) | fixed | no relocalisation where no map frame is near; a jump must be plausible for the elapsed time (0.5 m + 2 m/s) unless ≥ 300 inliers; selection radius 8 → 15 m → L2 0.30 m |
| 15 | ✗ CV: the challenger accepted a slowly drifting series of fixes (L5 under F2: 0.41 m) | fixed | compare with the challenger's first fix propagated by dead reckoning, 8 frames → 0.11 m |
| 16 | Remaining weak spots: lines no map lap drove (L2 warm-up line, pit exit of L0 in F4) | open | the test-run map uses all 9 laps; a map must cover the lines the car will drive |
| 17 | Rely more on car-state odometry (Exp23): lower fix weight, lower process noise, or lower fix weight only when fixes are weak / far from the map lines | – not adopted | globally worse (L5 0.094 → 0.18–0.23 m); weak-fix-only variants help hard laps (val 0.156 → 0.12–0.13 m, L2 0.30 → 0.21 m) but cost on the tuning lap L5, so not chosen; exposed that a wrong relocalisation can pass the ≥ 300-inlier bypass |

### 6.1 Why a filter, and why not retrieval every frame
Exp12 already gave 0.16 m, but it re-solved the full problem from scratch for every frame: retrieval, triangulation
of the map on the fly, two LightGlue matches per camera. That cost ~410 ms. It also had no memory, so one wrong
retrieval meant one 5 m error. A tracking system can do better on both counts, because between two frames we know
the motion very well from the car state (dead reckoning drifts only ~2 cm per frame).

**No visual odometry in the final system.** The prediction step uses only the allowed car-state signals (vx, vy, yaw
rate wz). The camera images enter only through the absolute map fixes. This follows from stage 1 (val, legacy split):

| motion source | RPE 1 s yaw | RPE 10 s drift / yaw | 2 laps, no fixes |
|---|---|---|---|
| car-state dead reckoning (Exp01) | **0.087°** | **0.57 % / 0.36°** | **20.5 m** |
| best camera-only VO (Exp05, planar 7-camera) | 0.34° | 1.25 % / 1.26° | 43.3 m |
| car state with gyro bias tracked by VO (Exp06) | 0.087° | 0.57 % / 0.37° | 20.6 m (no gain) |

So the car state is about 4× better than VO over 1–10 s, and VO did not even help to correct the car state's slow
drift. VO would also add a LightGlue match per camera between consecutive frames on top of the map matching. In the
final system, odometry mostly bridges only 0.1 s between two fixes (98.5–100 % of frames get an accepted fix on the
mapped CV laps), where dead reckoning errs by ~2 cm. VO would
matter only if the car state were unavailable or degraded (for example if the test run's speed or yaw-rate signals were
missing); then Exp05 is the camera-only fallback. Note the caveat from section 7 (question 5): we do not know whether
the car-state velocities are themselves GNSS-aided.

### 6.2 Offline: a cached map of the training run (Exp20)
For every training frame of every camera we extract SuperPoint keypoints, match them with LightGlue to the previous
and next frame (≥ 1.5 m apart), and triangulate them with the training GT. We store, per point, the pixel, the
256-D descriptor and the 3D position in `map`: 20 M points, 10.6 GB, loaded on the GPU once. This is the "training
run becomes a map" idea (method 3a of the slides), done once instead of per query. For cross-validation, a fold's
map is the same data restricted to that fold's laps; for the hidden test run, all 9 laps are used.

### 6.3 Online, per frame (Exp21)
1. **Predict** the pose with the car state (vx, vy, wz), causally. No visual odometry (see 6.1).
2. **Select** map frames: the nearest frame (within 15 m, similar heading) from each of the 3 closest passes of the
   map laps, preferring frames whose GT was not near an accuracy spike.
3. **Guided matching.** Project those frames' 3D points into the 7 query images with the *predicted* pose. Each projected
   point only looks for a SuperPoint keypoint within ~24 px with a similar descriptor. Because the prediction is
   good, this replaces both retrieval and LightGlue and costs about a millisecond.
4. **PnP** over all cameras at once (known extrinsics). The unknowns are x, y, yaw and, with priors, small corrections
   to roll, pitch and height relative to the map frame. The car pitches under braking; ignoring that biased the position.
5. **Kalman update** with an outlier gate, plus two safety nets:
   * (a) **Challenger.** If 8 rejected fixes agree with the first of them, carried forward by dead reckoning, they take over.
   * (b) **Relocalisation.** If nothing is accepted for 1 s while map data is near, MegaLoc retrieval proposes places,
     which are verified with a wide-window PnP. A jump is accepted only if it is plausible for the time since the
     last fix, or if the evidence is very strong.

### 6.4 What we learned
* **One camera is not enough.** With only `fr` (narrow), a distant wall makes yaw and sideways translation nearly
  interchangeable; the tune lap reached 0.5 m. Adding the side and rear cameras removed this (0.12 m).
* **The ground truth is not uniformly good.** At one spot the GT's own `pos_accuracy` jumps to 2–3 m on most laps.
  Map frames recorded there carry the error into the map, so we avoid them when another lap is available. Using
  this GT metadata for the map is allowed (training data).
* **Pitch matters.** Estimating Δpitch/Δroll (with priors) reduced lateral error by ~10 %.
* **The error floor.** On well-mapped laps all further parameter changes stay within 0.103–0.108 m. The residual has no
  bias and repeats at the same corners on every lap: it is set by the GT (~0.1 m) and by inconsistencies between the
  laps that form the map, not by the matcher. The largest remaining errors (up to 0.45 m) are in hard left turns
  (lateral acceleration ~5 m/s²) and at a corner exit under acceleration (`outputs/21_online/val/high_error_moments.md`).
* **Runtime.** The first version took 150 ms. The cost was not the neural network but small numerical operations:
  a Gauss-Newton solver on the GPU in float64 (launch-bound) and repeated GPU→CPU synchronisations. An analytic
  Jacobian on the CPU and batched matching brought the full 7-camera pipeline to **40 ms**, including JPEG decoding.
* **Safety nets need the same skepticism as measurements.** Both bugs found by cross-validation (14, 15 in the
  table) were in the recovery logic, not in the matching. Each accepted a jump that dead reckoning said was
  impossible. Both now check consistency with dead reckoning before overriding the filter.
* **Coverage of the map is what limits accuracy.** Where the car drives a line that no map lap drove (L2's warm-up
  line, up to 14 m off; the pit exit of L0 when L0 is not in the map) matches are sparse or absent and the error
  grows to 1.7–2.4 m. Everywhere else it is 0.07–0.12 m.

### 6.4b Relying more on car-state odometry (Exp23)
Question: since dead reckoning is so smooth, would trusting it more (and the camera fixes less) lower the error?
Seven variants were run on every validation lap of F1–F4 and on L5 (`outputs/23_odo_weight/README.md`).
* **Globally, no.** Raising the fix noise floor to 0.2 / 0.4 m roughly doubles the error on L5 (0.094 → 0.18 / 0.23 m).
  Odometry drifts (speed scale, yaw-rate bias); the fixes are better than odometry beyond a fraction of a second.
* **An overconfident filter breaks.** Halving the process noise made the filter reject correct fixes, declare itself
  lost, and accept an 18 m wrong relocalisation that had 328 inliers (above the 300-inlier "strong evidence" bypass).
* **Selectively, yes on hard laps.** Down-weighting fixes with few inliers, or fixes taken far from the mapped lines,
  improves the 6 validation laps from 0.156 to 0.12–0.13 m and L2 from 0.30 to 0.21 m (peak below 1 m). But on the
  well-mapped tuning lap L5 it costs 0.017–0.026 m, so by the protocol it is not adopted. A cross-track-only or
  delayed inflation, chosen with held-out-lap tuning, is the next step.

### 6.5 Track borders (Exp22): tried, dropped
The surveyed borders project exactly onto the painted edge lines with the GT pose, so the idea is sound. But the
cameras sit 0.34–0.42 m above the asphalt and see it at a grazing angle: moving the car 0.5 m sideways moves the
projected border by only ~1 px (median), below the 4–6 px noise of edge detection, and ~80 % of edge samples lock on to
other edges (the other side of the white line, kerb stripes). The update made the pose worse. A dedicated line
segmentation might rescue it, but geometry would still limit it to roughly ±0.3 m.

### 6.6 Results

**Cross-validation over laps (current protocol, `docs/laps.md`, full table in [`cv_performance.md`](cv_performance.md)).**
Each fold builds its map from 5–6 laps and validates on 2–3 others; L5 is always the tuning lap and never in a map.
Every lap is run as its own segment from its first GT pose, with parameters frozen.

| fold | validation laps | pos RMSE | median | lateral | yaw | ≤ 1 m |
|---|---|---|---|---|---|---|
| F1 | L1 / L6 | 0.080 / 0.083 m | 0.059 / 0.062 m | 0.043 / 0.057 m | 0.055 / 0.074° | 100 / 100 % |
| F2 | L2 / L7 | 0.302 / 0.089 m | 0.110 / 0.062 m | 0.272 / 0.057 m | 0.254 / 0.088° | 97.5 / 100 % |
| F3 | L3 / L4 | 0.068 / 0.117 m | 0.052 / 0.074 m | 0.043 / 0.074 m | 0.059 / 0.103° | 100 / 100 % |
| F4 | L0 / L4 / L7 | 0.686 / 0.117 / 0.088 m | 0.187 / 0.071 / 0.059 m | 0.601 / 0.071 / 0.057 m | 0.470 / 0.101 / 0.083° | 87.2 / 100 / 100 % |
| **F1–F3 pooled (6 laps, each once)** | | **0.156 m** | **0.066 m** | **0.131 m** | **0.132°** | **99.5 %** |

Runtime ~41 ms per 7-camera image set. L4 was a development lap under the legacy split, so its numbers are slightly
optimistic. The relocalisation bug (14) was discovered on a validation lap; its fix was tuned on L5 only.

**Legacy split** (val = last two laps ≈ L6 + L7, earlier parameter set): 0.097 m RMSE, median 0.059 m, lateral
0.067 m, yaw 0.094°, 100 % within 0.5 m, 40.7 ms per frame (Exp12: 0.160 m, 410 ms). In test mode on a lap
packaged as a separate run (only its first pose given): 0.086 m with the official script.

Robustness checks: gamma 0.5/2, gain 0.4, noise and blur on the query images change the RMSE from 0.104 m to at most
0.119 m; injected pose errors of 3–80 m are recovered in 1–2 s.

### 6.7 Remaining risks and next steps
* **Lighting.** A daylight test run against a night map is untested. Photometric tests are only a proxy. If matching
  fails, the filter falls back on dead reckoning (about 10 m drift per lap) and relocalises when matching returns.
  Mitigation if test-like data appear: add those images to the map (the map builder takes any run with poses).
* **Lines not in the map.** A racing line far from every map lap (as L2) or an unmapped pit section costs 0.3–0.7 m
  RMSE on that lap. The test-run map uses all 9 laps, including the pit exit (L0) and entry (L8).
* **Hard corners.** The largest errors on well-mapped laps are at high lateral acceleration (outward lateral error,
  heading lag). Next to try, on L5: looser roll/pitch priors when |ay| is large, and per-camera weights in turns.
* **GT quality at the score.** At the GT-spike spots the score depends on the GT's own error there.

---

## 7. Questions for the dataset provider

These are the 10 questions that matter most, ordered by how much the answer could change our method or our
score. They were checked against the INTRO slides, the dataset README and the challenge description
([`Constructor Labs - Camera-based localization on a race track.pdf`](Constructor%20Labs%20-%20Camera-based%20localization%20on%20a%20race%20track.pdf)).
Questions those documents already answer have been dropped (listed at the end).

1. **Using the training run as a map at test time.** Our best method builds 3D points by triangulating training
   images with their **training GT poses**, and uses them at inference on the test run. The rules say GT is "only used
   for training and for validation" and is not allowed "at inference". We read this as banning *test-run* GT only;
   slide 17 ("training run becomes a map") suggests the same. Can you confirm this is allowed? May the code read the
   training mcap during the test run, or should the map be shipped as a precomputed file? Is there a size limit?
2. **Same track, frame and car in the test run.** Is the hidden test run on the same layout as the training run (Yas Marina
   North), in the same driving direction and the same `map` frame? Is it the same car, with the same camera mounts and
   calibration (`camera_info`, `/tf_static`)?
3. **Test-run conditions.** "Different lighting" — does that mean daylight or dusk (sun, shadows), or night under different
   lights? How much faster is it? Does it include the pit lane, stops or other cars? Could you share **any short
   daylight clip from the same track** (with or without GT) for us to check how robust the appearance matching is?
4. **Car-state fields in the test run.** The description says the dataset has "no IMU data", but `state_estimation`
   contains `wz`, `ax`, `ay` (and `valid_imu_b` is `False` all run). Will the test mcap contain `state_estimation` with
   the same fields? Which fields are explicitly allowed beyond speed, accelerations and yaw rate: wheel speeds
   `omega_w_*`, steering `delta_wheel_rad`, sideslip `beta_rad`, `vx`, `vy`?
5. **Are `vx`, `vy`, `v_mps` and `wz` GNSS-aided** (outputs of the same fusion as the GT pose), or raw wheel / gyro
   signals? Integrating them alone gives 20 m error over 2 laps. If they are GNSS-aided, a car in a real GNSS outage
   would not have signals this good.
6. **Image time stamps.** The 7 stamps agree to within 0.025 ms, yet we measure a consistent **−9.1 ms** offset between
   the images and the GT/state clock (0.36 m at 40 m/s). Is `header.stamp` the trigger time, the start of exposure or
   the middle of exposure? What is the exposure time at night (fixed or auto)? Are the sensors rolling-shutter, and what
   is the readout time? Are cameras and the state estimator on the same PTP-synchronised clock?
7. **GT time semantics.** Is the GT stamp the time the pose is valid for, or the time it was published? Is the GT
   the online estimate or a smoothed / post-processed trajectory, and will the test GT be produced the same way?
8. **GT uncertainty and scoring.** Our val error (0.16 m) is close to the stated GT accuracy (~0.1 m / 0.2°).
   Is `pos_accuracy.*` the per-sample GT uncertainty? Its median on val is 0.30 m, with spikes up to 300 m. What do
   `se_status` and `se_state` mean? Are low-quality GT intervals (and standing still or the pit lane) excluded from the
   test score?
9. **Anisotropic intrinsics.** `fr`, `cl`, `cr` and `rr` have fx/fy ≈ 1.15 (for example `cr`: fx 744, fy 644), while `fl`, `r` and
   `rl` have fx ≈ fy. Were these images resized from another sensor resolution, or do the K matrices belong to a
   different image size? Our VO also finds `fr` and `cr` under-report rotation by 6.5 % and 12 %. Is a newer intrinsic
   calibration available?
10. **Extrinsic accuracy.** Roll and translation are "mount values" and `cl`/`cr` keep an older calibration. Can you
    give error bounds, or the 2026-10-12/14 calibration for `cl`/`cr`? Are the cameras rigid with respect to the chassis
    under braking, kerb strikes and aero load? Our VO errors correlate with these events.

**Already answered by the challenge description, so dropped:** runtime scoring (5 %, full points at ≤ 100 ms per frame, self-reported
with hardware); score weights (position 25 %, lateral 20 %, heading 10 %, ≤ 1 m 10 %); only the first test pose is given;
the validation laps are ours to choose; pre-trained models are allowed but must be declared; the submission is x, y, yaw
(option B, s/n, appears only in the INTRO slides, so the racing-line file question was dropped). Lower priority and
also dropped: what exactly the surveyed borders represent (painted line, kerb or barrier), which matters only once
stage 3 starts.
