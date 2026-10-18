# Laps of the training run and the cross-validation split

Run time t is in seconds since the first GT stamp (19:02:37.114 UTC, 2026-10-13). A lap starts at the start-line
crossing. Lap boundaries are defined in `odo/data.py` (`LAPS`); the folds are in `FOLDS`.

## Laps

| lap | kind | t start [s] | t end [s] | start (UTC) | lap time | distance | v max [m/s] | v mean [m/s] | slow stretches (s into the lap) |
|---|---|---|---|---|---|---|---|---|---|
| L0 | out-lap | 0.0 | 185.5 | 19:02:37.114 | 185.5 s | 1,649 m | 14.9 | 8.9 | 80 s in the pit lane |
| L1 | warm-up | 185.5 | 365.6 | 19:05:42.570 | 180.1 s | 2,979 m | 30.1 | 16.5 | 0–44 s, 62–75 s |
| L2 | warm-up | 365.6 | 523.4 | 19:08:42.683 | 157.8 s | 3,037 m | 28.3 | 19.2 | 26–65 s, incl. a 4.0 s stop |
| L3 | flying | 523.4 | 634.0 | 19:11:20.489 | 110.6 s | 2,980 m | 35.5 | 26.9 | none (reference lap) |
| L4 | partial | 634.0 | 752.2 | 19:13:11.086 | 118.2 s | 2,980 m | 35.4 | 25.1 | 73–93 s, 105–118 s |
| L5 | partial | 752.2 | 896.1 | 19:15:09.306 | 143.9 s | 2,982 m | 35.4 | 20.6 | 0–17 s, 33–50 s |
| L6 | flying | 896.1 | 1007.7 | 19:17:33.230 | 111.6 s | 2,981 m | 40.3 | 26.6 | 23–34 s, incl. a 0.9 s stop |
| L7 | flying | 1007.7 | 1120.7 | 19:19:24.787 | 113.0 s | 2,978 m | 40.3 | 26.2 | none (0.3 s stop) |
| L8 | in-lap | 1120.7 | 1194.4 | 19:21:17.817 | 73.7 s | 1,257 m | 30.3 | 17.0 | 28 s in the pit lane |

L9 is the new data in data/ without ground truth - rosbag_2026-10-13_16-08-46_cameras_lap.

## Cross-validation folds (current protocol)

| fold | map | laps in the map (training) | validation laps | tuning lap |
|---|---|---|---|---|
| F1 | `map_cv_f1` | L0 L2 L3 L4 L7 L8 | L1, L6 | L5 |
| F2 | `map_cv_f2` | L0 L1 L3 L4 L6 L8 | L2, L7 | L5 |
| F3 | `map_cv_f3` | L0 L1 L2 L6 L7 L8 | L3, L4 | L5 |
| F4 | `map_cv_f4` | L1 L2 L3 L6 L8 | L0, L4, L7 | L5 |
| F5 | `map_cv_f2_swap` | L0 L1 L3 L4 L6 L8 | L5, L7 | L2 |
| F6 | `map_cv_f1_swap` | L0 L5 L3 L4 L7 L8 | L1, L6 | L2 |
| F7 | `map_all` | night L0-L8 | day L2 (day proxies) | night L5 |

* In F1–F4, L5 is the tuning lap: it is the only lap used to choose parameters, and it is in none of their maps.
* In F1–F3 each of L1, L2, L3, L4, L6 and L7 is a validation lap exactly once; pooling F1–F3 gives a result over
  6 laps. F4 adds the out-lap L0 (80 s in the pit lane) as a validation lap and re-validates L4 and L7 with a smaller
  map (5 laps, without L0, L4, L5, L7). L8 (in-lap) is in every map.
* A map `map_cv_fN` is the cached feature map (`outputs/20_map`) restricted to frames whose own stamp and whose
  triangulation partners lie in that fold's map laps. No rebuild is needed.
* F5 is F2's map with the roles of L2 and L5 swapped; F6 is F1's map with L2 replaced by L5. Both use L2 as the
  tuning lap and put L5 in the map or the validation set, so for them "L5 is never in a map" does not hold. The
  parameters were still chosen on L5, so L5 is not an unseen lap for F5/F6. Results: [`cv_performance.md`](cv_performance.md).
* F7 is the test-run setting for daylight: the map is all night laps, and the validation lap is the day proxy of L2
  (`scripts/day_translate.py`). L2's night frames are in the map, so F7 isolates the night→day appearance change at
  identical viewpoints; it is an upper bound. Results: `outputs/24_day/summary.md` (day 0.066 m, day_nolights 0.067 m).
* For the hidden test run, the map is the whole training run (L0–L8).
