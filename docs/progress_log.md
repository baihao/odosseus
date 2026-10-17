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
