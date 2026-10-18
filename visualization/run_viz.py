#!/usr/bin/env python3
"""Real-time visualization of a localiser run (odo.online): overhead track view of the estimate (and the ground
truth), live errors or health signals, the joint camera fix per frame and running metrics, as an mp4 at 10 fps
and/or a window paced at the 10 Hz image-set rate.

    # mapped: a lap with ground truth, e.g. a cross-validation run (python scripts/exp21_cv.py F5)
    python visualization/run_viz.py mapped outputs/21_online/cv/F5/L7                # whole lap -> mp4
    python visualization/run_viz.py mapped outputs/21_online/cv/F5/L2 --t0 60 --t1 90 --live

    # test: a run without ground truth, e.g. ./run.sh <test.mcap> outputs/submission --first_pose
    python visualization/run_viz.py test outputs/submission                         # images from <run>/extracted

Ground truth is used only for the errors, the metrics and the drawn true path. Outputs go to
visualization/outputs/<name>/ (gitignored): <name>.mp4, final_frame.png, plots/<panel>.png, metrics.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as ref  # noqa: E402
from live_metrics import RunningMetrics  # noqa: E402
from odo_run import ACCEPTED, GATED, Day, Night, load_run  # noqa: E402
from reference import data  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
THUMB_CAM = 'fr'


def figure(interactive):
    """16:9 figure: 1920x1080 for the mp4, a smaller window on screen (same layout, other pixel density)."""
    if interactive:
        import matplotlib.pyplot as plt
        return plt.figure(figsize=(16, 9), dpi=80)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(16, 9), dpi=120)
    FigureCanvasAgg(fig)
    return fig


def write_poses(d, path):
    with open(path, 'w') as f:
        f.write('timestamp_ns,x,y,yaw\n')
        for t, x, y, a in zip(d['stamp_ns'], d['x'], d['y'], d['yaw']):
            f.write(f'{int(t)},{x:.4f},{y:.4f},{a:.6f}\n')


def chunk_text(args):
    if args.t0 is None and args.t1 is None:
        return 'whole run'
    return f'{args.t0 or 0:g}–{args.t1:g} s' if args.t1 is not None else f'from {args.t0:g} s'


def show(dash_cls, d, title, subtitle, out, name, args, summary):
    import player
    if not args.no_mp4:
        dash = dash_cls(d, figure(False), title, subtitle)
        summary['mp4'] = player.mp4(dash, os.path.join(out, f'{name}.mp4'))
        dash.save_images(out)
        print(f'mp4: {summary["mp4"]["frames"]} frames in {summary["mp4"]["render_s"]:.0f} s -> {out}/{name}.mp4')
    if args.live:
        summary['live'] = player.live(dash_cls(d, figure(True), title, subtitle), d['stamp_ns'], args.speed,
                                      hold=not args.no_hold)
        print('live window:', json.dumps(summary['live']))
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=float)


def mapped(args):
    t = time.time()
    run = args.run.rstrip('/')
    d = load_run(run, data.allowed_state(), None if args.no_thumbs else data.frames(THUMB_CAM), args.t0, args.t1)
    lap = d['cfg'].get('split', '?')
    fold = args.fold or next((c for c in run.split('/') if c in data.FOLDS), None)
    rm = RunningMetrics(ref.ground_truth(), d['stamp_ns'], d['x'], d['y'], d['yaw'], d['proc_ms'], d['n_acc'])
    d.update(gt_x=rm.e['gt_x'], gt_y=rm.e['gt_y'], gt_yaw=rm.gt_yaw, lat=rm.e['lat'], lon=rm.e['lon'],
             pos=rm.e['pos'], dyaw_deg=rm.e['dyaw_deg'], ok=rm.ok, running=[rm.at(k) for k in range(len(d['t']))],
             bounds=ref.track_bounds(), ref_xy=(rm.gt['x'], rm.gt['y']))
    chunk = rm.final()                                   # = tools/evaluate.py math; asserts the running values agree
    name = args.name or f'mapped_{fold + "_" if fold else ""}{lap}_' + chunk_text(args).replace(' ', '').replace('–', '-')
    out = os.path.join(OUT, name)
    os.makedirs(out, exist_ok=True)
    write_poses(d, os.path.join(out, 'chunk_poses.csv'))           # the official scorer on the same chunk
    ref.write_gt_csv(rm.gt, os.path.join(out, 'chunk_gt.csv'))
    official = ref.metrics.run_official(ref.OFFICIAL_EVAL, os.path.join(out, 'chunk_gt.csv'), os.path.join(out, 'chunk_poses.csv'))
    with open(os.path.join(out, 'official_evaluate_chunk.txt'), 'w') as f:
        f.write(official)
    print(official)
    maps = f'map: night laps {" ".join(data.FOLDS[fold]["map_laps"])}' if fold else 'map: see the run'
    print(f'{lap} ({chunk_text(args)}): {len(d["t"])} frames, pos {chunk["pos_rmse_m"]:.3f} m, lateral '
          f'{chunk["lat_rmse_m"]:.3f} m, heading {chunk["yaw_rmse_deg"]:.3f} deg, <1 m {chunk["within_1.0_pct"]:.1f} %; '
          f'{maps} (prepared in {time.time() - t:.0f} s)')
    title = f'Mapped mode · lap {lap}, {chunk_text(args)} · {maps}'
    summary = {'run': run, 'lap': lap, 'fold': fold, 'chunk': chunk_text(args), 'frames': len(d['t']),
               'metrics': chunk, 'runtime_ms': {'mean': float(d['proc_ms'].mean()), 'p95': float(np.percentile(d['proc_ms'], 95))},
               'fix_frames_pct': float(100 * np.isin(d['status'], ACCEPTED).mean()),
               'speed_kmh': [float(d['speed_kmh'].min()), float(d['speed_kmh'].max())]}
    show(Night, d, title, None, out, name, args, summary)


def test(args):
    run = args.run.rstrip('/')
    query = args.query or os.path.join(run, 'extracted')
    q = data.Run(query)
    d = load_run(run, q.allowed_state(), None if args.no_thumbs else q.frames(THUMB_CAM), args.t0, args.t1)
    line = ref.racing_line()
    d['has_ref'] = line is not None
    if line is not None:
        s, n, _, _ = line.xy_to_sn(d['x'], d['y'], t_ns=d['stamp_ns'])
        d.update(ref_s=s, ref_off=n, ref_xy=(line.p[:, 0], line.p[:, 1]))
    else:
        print(f"organizers' racing line not found ({ref.UPDATE}): no offset panel values")
        d.update(ref_off=np.zeros(len(d['t'])), ref_xy=(np.zeros(0), np.zeros(0)))
    d['bounds'] = ref.track_bounds(query)
    cfg = d['cfg'].get('cfg', {})
    standin = os.path.exists(os.path.join(run, 'first_pose_standin.json'))
    d['start_note'] = ('stand-in' if standin else 'given first pose') + \
        f', σ {cfg.get("init_sigma_xy", 0.05):g} m / {cfg.get("init_sigma_yaw_deg", np.degrees(0.002)):.2g}°'
    name = args.name or f'test_{os.path.basename(run)}_' + chunk_text(args).replace(' ', '').replace('–', '-')
    out = os.path.join(OUT, name)
    os.makedirs(out, exist_ok=True)
    acc = np.isin(d['status'], ACCEPTED)
    off = d['ref_off']
    summary = {'run': run, 'query': query, 'chunk': chunk_text(args), 'frames': len(d['t']), 'start_pose': d['start_note'],
               'fix_frames_pct': float(100 * acc.mean()), 'gated_pct': float(100 * np.isin(d['status'], GATED).mean()),
               'relocalisations': int(np.isin(d['status'], ('reloc', 'reset')).sum()),
               'sigma_xy_m': {'median': float(np.median(d['sigma_xy'])), 'max': float(d['sigma_xy'].max())},
               'runtime_ms': {'mean': float(d['proc_ms'].mean()), 'p95': float(np.percentile(d['proc_ms'], 95))},
               'coverage_pct': float(d['coverage'][-1]),
               'racing_line_offset_m': ({'rms': float(np.sqrt(np.mean(off ** 2))), 'p95_abs': float(np.percentile(np.abs(off), 95))}
                                        if d['has_ref'] else None),
               'cam_matches_median': dict(zip(d['cams'], np.median(d['cam_matches'], 0).tolist())),
               'speed_kmh': [float(d['speed_kmh'].min()), float(d['speed_kmh'].max())]}
    print(json.dumps({k: v for k, v in summary.items() if k not in ('cam_matches_median',)}, default=float))
    title = f'Test mode · {os.path.basename(run)}, {chunk_text(args)}'
    subtitle = (f'No ground truth: health signals only  ·  {len(d["cams"])} cameras, joint PnP to the map + car-state EKF'
                f'  ·  {d["start_note"]}')
    show(Day, d, title, subtitle, out, name, args, summary)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='source', required=True)
    n = sp.add_parser('mapped', help='a run of a lap with ground truth (odo.online output folder)')
    n.add_argument('run', help='folder with pred.csv, online_info.csv, config.json (e.g. outputs/21_online/cv/F5/L7)')
    n.add_argument('--fold', choices=list(data.FOLDS), help='the run\'s map fold (default: from the path)')
    te = sp.add_parser('test', help='a run without ground truth (e.g. the out_dir of ./run.sh)')
    te.add_argument('run', help='folder with pred.csv, online_info.csv, config.json')
    te.add_argument('--query', help='the extracted run (images, car state); default <run>/extracted')
    for p in (n, te):
        p.add_argument('--t0', type=float, help='chunk start, s after the first pose (default: start)')
        p.add_argument('--t1', type=float, help='chunk end, s after the first pose (default: end)')
        p.add_argument('--no-thumbs', action='store_true', help='no camera thumbnail (skips reading images)')
        p.add_argument('--live', action='store_true', help='show a window at the image-set rate')
        p.add_argument('--speed', type=float, default=1.0, help='live speed (1 = real time)')
        p.add_argument('--no-hold', action='store_true', help='close the window at the end')
        p.add_argument('--no-mp4', action='store_true')
        p.add_argument('--name', help='output folder name under visualization/outputs/')
    args = ap.parse_args()
    {'mapped': mapped, 'test': test}[args.source](args)


if __name__ == '__main__':
    main()
