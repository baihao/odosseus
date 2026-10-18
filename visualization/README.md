# Visualization

A 10 Hz dashboard of a localiser run (`odo.online`): the overhead track view, a chase view, the camera image, live
errors (or health signals without ground truth), the cameras in the joint PnP fix per frame and running metrics.
It renders an mp4 at 10 fps (every frame) and, with `--live`, plays the run in a window at the image-set rate.

## Modes

```bash
# mapped: a lap with ground truth, e.g. from the cross-validation (python scripts/exp21_cv.py F5)
python visualization/run_viz.py mapped outputs/21_online/cv/F5/L7
python visualization/run_viz.py mapped outputs/21_online/cv/F5/L2 --t0 60 --t1 90 --live

# test: a run without ground truth, e.g. ./run.sh <test.mcap> outputs/submission --first_pose
python visualization/run_viz.py test outputs/submission
```

* **mapped** (`TruthDashboard`): ground truth and estimate on the track, position, lateral and heading error
  over time (with the filter's ±1σ), and the running official metrics. The chunk's poses and GT are written next to
  the video, and the dataset's own `tools/evaluate.py` is run on them (`official_evaluate_chunk.txt`).
* **test** (`HealthDashboard`): no ground truth. The organizers' racing line is drawn as context, and the panels show
  the offset from it, the filter's position σ, the map matches per camera and the health metrics (fix rate,
  relocalisations, runtime, coverage).

Options: `--t0 / --t1` (chunk in s after the first pose; default: the whole run), `--live`, `--speed`,
`--no-mp4`, `--no-thumbs`, `--name`; test mode: `--query` (the extracted run, default `<run>/extracted`).

## Inputs

* the run folder: `pred.csv`, `online_info.csv`, `config.json` (written by `odo.online`);
* mapped mode: the training run in `data/extracted` (GT, car state, images, track borders);
* test mode: the extracted test run (images, car state); the organizers' racing line from
  `data/23-00-38_cut_120m_update/` (`map.mcap`, `tools/track_frame.py`) if present;
* the official scorer `data/c0_2026-10-13_22-53-42_camloc/tools/evaluate.py` (mapped mode, optional).

Packages: those of `requirements.txt` (matplotlib, OpenCV, pandas). The mp4 is encoded with ffmpeg (H.264) when it
is installed, otherwise with OpenCV (mp4v).

## Outputs

`visualization/outputs/<name>/` (gitignored): `<name>.mp4`, `final_frame.png`, `plots/<panel>.png` (each panel of
the final frame), `metrics.json`; mapped mode also `chunk_poses.csv`, `chunk_gt.csv`, `official_evaluate_chunk.txt`.

## Files

| file | content |
|---|---|
| `run_viz.py` | command line: the two modes, outputs |
| `odo_run.py` | an `odo.online` run as per-frame arrays; the dashboards relabelled for the joint 7-camera PnP fix |
| `dashboard.py` | the figure: layout, panels, blitting; `TruthDashboard`, `HealthDashboard` |
| `live_metrics.py` | per-frame errors and running metrics with the official math |
| `reference.py` | GT, track borders, start/finish line, racing line, the official scoring math |
| `player.py` | the mp4 writer and the 10 Hz live window |
