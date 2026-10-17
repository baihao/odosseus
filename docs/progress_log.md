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
