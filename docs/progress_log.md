# Progress log

Format: `[day-hour-minute] description` (local time, 2026-10-17). Times are taken from the modification time of
each step's output or log file.

[17-02-53] Read task PDF + dataset README; started unzip of the mcap (7.0 GiB, 1194 s, 7 cams @10 Hz, GT /tf @106 Hz)
[17-02-54] Web search for VO SOTA: VGGT-Long / Pi-Long (KITTI-scale feed-forward), MASt3R-SLAM, VGGT-SLAM++, DPVO; cloned VGGT-Long + LightGlue
[17-02-57] scripts/extract_mcap.py -> data/extracted (GT, car state, calib, track bounds, 7 x ~11.9k raw JPEGs); val = last 2 full laps t=897..1120 s
[17-02-59] odo/ lib: data access, metrics (bit-exact with tools/evaluate.py), PDF/HTML/MP4 reports
[17-02-59] Exp01 dead reckoning (car-state vx,vy,wz): val pos RMSE 20.5 m, RPE1s 0.19 m / 0.09 deg - reference VO must beat
[17-03-05] Exp02 MASt3R-SLAM (r/fl/fr, calib & uncalib, relaxed thresholds): tracking lost after 1-3 frames -> rejected
[17-03-10] Installed VGGT-Long (+Pi3 backbone, loops off); 400-frame rear clip: VGGT RPE1s yaw 1.3 deg, Pi3 0.8 deg - both << gyro
[17-03-24] Exp04 own calibrated feature VO (SuperPoint+LightGlue, MAGSAC essential, 7 cams): median step yaw err 0.04 deg, ~15% outliers
[17-03-32] Exp04: side cams biased (rr -0.5 deg/step), per-camera rotation gain error (fr 0.935) -> gain/bias/sigma calibrated on tune lap
[17-03-34] Exp04 fusion: consensus gate locks in turns, KF lags at turn-in; final = per-step IRLS (Cauchy) over cameras
[17-03-43] Exp05 planar Ackermann 1-DoF multi-camera VO (metric step from speed, lever arm) + essential: 14 measurements, best on tune lap
[17-03-44] Exp06 gyro + VO bias tracking: no gain over dead reckoning on tune lap -> VO is not the bottleneck
[17-03-49] Exp03 Pi-Long full val on fr finished (~70 min)
[17-04-00] Stage 2: MegaLoc + DINOv2-GeM descriptors, 7 cams (db 8970 train frames, 2230 val); tune-lap sweep: 7-cam avg + top-10 -> 1.77 m RMSE
[17-04-04] Stage 1 val: DR 20.5 m | Pi-Long 313 m | Exp04 50.9 m | Exp05 43.3 m (best camera-only, RPE10s 1.25%/1.26 deg) | Exp06 20.6 m
[17-04-07] Stage 2 val: DINOv2-GeM 7 cams 1.70 m | MegaLoc fr top-1 1.59 m | MegaLoc 7 cams top-10 1.01 m (median 0.42 m)
[17-04-10] Exp12 retrieval + triangulated map frame (train GT) + 3-DoF multi-camera PnP: first 300 tune queries 0.168 m RMSE
[17-04-21] Exp12 full tune lap: median 0.098 m but 4.3% fallbacks; found image-vs-GT clock offset -9.1 ms (lon err ~ dt*(v_q - v_map))
[17-04-30] Exp12: geometric re-ranking over 3 distinct candidates + dt correction -> tune RMSE 0.73 m; added 7-camera retry
[17-04-55] Exp12 val (frozen params): pos RMSE 0.160 m, median 0.073 m, lat 0.105 m, yaw 0.184 deg, 100% within 1 m
[17-04-56] Summaries outputs/stage{1,2}_summary; docs/README.md; per-experiment READMEs in outputs/
[17-09-25] docs/experiments_explained.md: detailed write-up of stages 1-2 (for non-tracking readers) + 27 questions for the dataset provider
[17-09-50] Read challenge description PDF (docs/): test run may be faster + different light (sun); runtime 5 % (full at <=100 ms/frame); questions cut to the 10 most important
[17-10-54] Stage 3/4 start: odo/map_build.py - cached 3D feature map of the training run (SuperPoint per frame, LightGlue to prev/next frame, triangulated with train GT), 7 cams
[17-11-05] odo/online.py - real-time EKF: car-state dead reckoning (vx,vy,wz) + guided projection matching to the map + 3-DoF multi-camera PnP; fr only, 400 tune frames: 0.134 m, 48 ms
[17-11-06] Profiling: GPU float64 Gauss-Newton was launch-bound (17 ms/call) -> numpy PnP; ~24 ms/frame with 1 camera
[17-11-07] Track borders projected with GT pose match the painted edge lines (outputs/22_borders/overlay_*.jpg)
[17-11-12] Tune-lap failure t=714-750 s: all map passes but one agree on a 2 m offset; GT pos_accuracy spikes (2-3 m) at that spot on most laps incl. both val laps -> GT unreliable there
[17-11-15] Border edge alignment: 0.5 m lateral shift moves projected border ~1 px (cams 0.34-0.42 m above ground, grazing view) vs 4-6 px edge noise -> not usable as a precise cue
[17-11-27] Map frame selection prefers passes away from GT accuracy spikes; gated-fix challenger hypothesis resets the filter
[17-11-30] New dev lap dev2 = 752-897 s (map = train minus lap +-10 s); fr+cl: dev2 0.130 m, tune 0.41 m (slow lap, bad-GT spot)
[17-11-37] 6-DoF PnP (x, y, yaw + droll, dpitch, dz with priors, planar output): dev2 fr+cl 0.130 -> 0.119 m
[17-11-57] Runtime: analytic-Jacobian numpy PnP, sync-free batched guided matching, GPU JPEG decode -> 150 -> 54 ms (5 cams)
[17-11-58] Map complete: 7 cams, 11 816 frames each, 20 M triangulated points (10.6 GB)
[17-12-05] 7 cams + 6-DoF: tune 0.123 m / dev2 0.086 m; 5 cams 0.115 / 0.093; 7 cams 3-DoF 0.103 / 0.114
[17-12-11] Kidnap test: 3 m errors recover in 1-4 s, 10-25 m never -> needs global relocalisation
[17-12-19] Parameter sweep (floor, passes, thresholds, keypoints, priors, Q): all 0.103-0.108 m mean -> at the GT/map noise floor
[17-12-24] MegaLoc relocalisation when lost > 1 s (candidates verified by wide-radius PnP): 10 / 25 / 80 m kidnaps recover in 2.6 / 3.9 / 0.9 s
[17-12-27] Test-mode pipeline (data.Run, --split test --query, scripts/run_submission.sh); dev2 as fake test run: 0.086 m (official scorer)
[17-12-42] Photometric stress (gamma 0.5/2, gain 0.4, noise, 15 px blur, combo): 0.104 -> at most 0.119 m
[17-12-45] Runtime alone: 39.6 ms per 7-camera image set (p99 49 ms), H100 NVL + EPYC 9454; frozen params (floor 0.10 m / 0.23 deg)
[17-12-48] Exp21 val (frozen, once): pos RMSE 0.097 m, lat 0.067 m, yaw 0.094 deg, 100 % < 0.5 m, 40.7 ms/frame (Exp12: 0.160 m, 410 ms)
[17-12-48] outputs/stage3_summary (summary.pdf, overlay.html, metrics.csv); READMEs for outputs/20_map, 21_online, 22_borders
[17-13-02] Docs: evaluate.py scope - predictions cover only val (897-1120 s); coverage is over the GT file span, so val is scored with GT cut to val (outputs/21_online/val/gt_val.csv)
[17-14-44] Val high-error analysis (outputs/21_online/val/high_error_moments.md): top 5 peaks = 3 places, repeated on both laps (corner exit +0.22 m lon; left turns at ay~5 m/s2, up to 0.45 m lateral outward)
[17-15-54] New protocol: laps L0-L8 (docs/laps.md as table), 3-fold lap CV F1-F3 (+F4 added by user), L5 = tuning lap; odo/data.py LAPS/FOLDS, online --fold
[17-15-54] First CV run: F1 0.08, F3 0.07-0.12, but F2/L2 3.7 m (false MegaLoc relocalisation, 18 m jump while off the mapped lines) and F2/L5 0.41 m (drifting challenger)
[17-15-54] Fixes tuned on L5 only: no reloc when no map nearby, plausibility gate for reloc jumps (0.5 m + 2 m/s, or >= 300 inliers), selection radius 15 m, fixed-seed challenger with 8 frames; F2/L5 0.41 -> 0.11 m; kidnaps still recover in 1-2 s
[17-15-54] Final CV (frozen): F1-F3 pooled over 6 laps 0.156 m (median 0.066, lat 0.131, yaw 0.132 deg, 99.5 % < 1 m); per lap 0.07-0.12 m except L2 0.30 m (off-line lap); F4/L0 0.69 m (pit exit unmapped); 41 ms alone
[17-16-01] docs/cv_performance.md: performance table per fold and validation lap (scripts/cv_table.py)
[17-16-11] docs/experiments_explained.md: section 6 rewritten (decision shortlist, CV results, recovery-logic lessons), data split section moved to the lap protocol
[17-16-25] Docs: final system uses no VO (car state ~4x better than best VO over 1-10 s; Exp06 VO-aided gyro no gain) - stated in experiments_explained 6.0/6.1 and README
[17-16-37] docs/experiments_explained.md: added 'What tuning means here' (definition, tuning lap L5, what was tuned per stage)
[17-16-49] docs/precomputed_data.md: sizes/locations of precomputed data (map 10.6 GB, 19.9 M points; MegaLoc 1.4 GB; both git-ignored)
[17-17-05] Folds F5 (F2 map, L2/L5 roles swapped) and F6 (= F1) run: F5 reproduces F2 (L5 0.111, L7 0.089, L2 0.302 m), F6 reproduces F1 within 1 mm; docs/cv_performance.md, laps.md updated
[17-17-55] F6 redefined by user (F1 map with L2->L5, tune L2) and rerun: L1 0.082, L6 0.080, L2 0.154 m (vs 0.302 under F2 map); F5 per-frame error plot docs/figures/F5_per_frame_error.png (scripts/plot_per_frame.py)
[17-18-12] F5/L2 diagnostics figure (docs/figures/F5_L2_diagnostics.png, scripts/plot_lap_diagnostics.py): error peak = 8.5 m off the map lines + inlier collapse in all camera groups + MegaLoc top-1 10 m off; corr(err, dist to map) 0.49
[17-19-03] F5/L2 lateral vs longitudinal error plot (docs/figures/F5_L2_lat_lon.png): lateral = 81 % of squared error, lon max 0.60 m
[17-19-04] outputs/21_online/cv/F5/L2/report_L2.pdf: appended lateral/longitudinal and diagnostics pages (6 pages)
[17-19-37] Exp23 (outputs/23_odo_weight): relying more on car-state odometry - global down-weighting of fixes worse (L5 0.094 -> 0.18-0.23 m); halved process noise -> overconfident filter, 18 m false reloc on L2 (328 inliers > 300 bypass); weak-fix-only variants (rinl400, rdist3) help val (0.156 -> 0.134/0.123 m, L2 0.21 m) but cost on L5 -> not adopted
[18-09-43] Exp24 (night -> day transfer): day proxy = CycleGAN-Turbo night_to_day (img2img-turbo, BDD100k) on the query images, env daytrans, scripts/day_translate.py
[18-10-33] Day proxy of L2 (all 7 cams); harder variant day_nolights (lamps/lit windows >= 200 removed before translation, as in daylight) at 18-10-49; only 4-13 % of map points lie on light sources
[18-10-39] Night F5/L2: masking query keypoints on light sources -> 4.2 m (lamps carry along-track information) -> not used
[18-10-52] Unchanged system on day proxies, F5/L2 (map = night laps): day 0.200 m, day_nolights 0.220 m vs night 0.302 m; 97-98 % frames with a map fix, PnP inliers 65-73 % of night
[18-10-51] odo/run_report.py: per-run PDF (pos/lat/lon error, innovation fix - dead-reckoning prediction, fix rate, global-retrieval attempts/fixes, inliers, zoom on the worst error); online.py logs innov*, reloc_try
[18-11-00] Stress on day_nolights: keep 50/25 % keypoints 0.204/0.227 m; keep 10 % 2.46 m; gain 1.6 (overexposed) 1.24 m; gamma 1.8 0.232 m
[18-11-13] Failure 1 (gain 1.6): a 129-inlier relocalisation jumped 2.5 deg in yaw after 1 s of gating -> 7 m drift. Reloc yaw plausibility 0.3 deg + 0.1 deg/s (DR yaw drift ~0.09 deg/s): gain16 1.24 -> 0.46 m; night L2, L5, day proxies, kidnaps unchanged
[18-11-27] Failure 2 (keep 10 %): a 30-inlier fix moved yaw 1.6 deg -> correct fixes gated, 11 m drift. Weak fixes (< 60 inliers) now update position only: keep10 2.46 -> 0.39 m, day_nolights 0.220 -> 0.190, night L2 0.302 -> 0.299, L5 unchanged; 100 inliers hurts night L2 (0.61 m)
[18-11-27] run_report.py also writes outputs/<run>/summary.md (setup, cfg diff, metrics vs reference, fix stats, worst error episodes)
[18-11-56] Untouched lap F5/L7: day_nolights 0.082 m vs night 0.089 m (100 % map fixes, 67 % of night inliers); final config = Exp21 there; MegaLoc retrieval on day proxies: best-of-4 within 15 m 100 % (night 99.4 %)
[18-11-56] New defaults (reloc_yaw0_deg 0.3, reloc_yaw_rate_deg 0.1, yaw_min_inl 60); kidnaps still recover in 1.0-3.4 s; outputs/24_day/summary.md, docs/final_method.md
[18-12-20] F7 (map = all night laps, val = day proxy of L2, final config): day 0.066 m, day_nolights 0.067 m vs night self-matching floor 0.067 m; 58-63 % of night inliers, 100 % map fixes (upper bound: same viewpoints as the map)
[18-12-36] Better day proxy: Qwen-Image-Edit-2509 (20B, Apache-2.0) + Lightning 4-step LoRA, instruction "midday sun, all lights off, geometry unchanged" (env qwenedit, scripts/day_translate_qwen.py); realistic sun, shadows, lamps off; 1.3 s/image
[18-12-41] Qwen output is shifted/zoomed vs the source (median 13.6 px; 7.9 px with 32-px-aligned sizes) -> scripts/register_proxy.py warps each output onto its night frame with one LightGlue homography: 2.9 px residual (local warping, CycleGAN 1.5 px); full L2 translation started (~4 h), -> data/day_proxy/L2_qwen_raw -> L2_qwen
[18-13-52] Qwen day proxy reduced to L2, cameras cl + cr (localisation with cl + cr only); CAP-VSTNet style transfer tried as an alternative (1.5 px alignment, but dusky blue look), not used
[18-13-52] min_inl 30 -> 15 (F5/L2, cl + cr): night unchanged 0.287 m, day_nolights 0.265 -> 0.255 m; with 10 % keypoints more fixes but noisier (night 0.46 -> 0.51 m, day 0.80 -> 0.81 m); no blow-ups, default kept at 30 (outputs/24_day/summary.md)
