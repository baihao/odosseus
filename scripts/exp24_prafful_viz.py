"""Prafful's dashboard (odosseus_prafful/visualization, unchanged) on runs of our final method (Exp24, odo.online).

    python scripts/exp24_prafful_viz.py night outputs/24_day/night/final/F5_L2 --lap L2 [--t0 0 --t1 30]
    python scripts/exp24_prafful_viz.py day outputs/24_day/L9/final --query data/L9/extracted

night: TruthDashboard (GT of the training run: errors, official metrics); day: HealthDashboard (L9, no GT).
Output: <run>/prafful_viz/<name>/ (<name>.mp4, final_frame.png, metrics.json).

What differs from prafful's camloc pipeline, and how it is mapped onto his panels:
* our fix is ONE joint PnP over all cameras (guided matching to the night map), not one fix per camera. "Camera
  fixes per frame" = cameras with PnP inliers in the joint fix (accepted by the filter / rejected at the gate);
  the camera bars = stage-1 guided matches to the night map per camera (the fix needs >= 30 inliers in total).
* his localization/artifacts are not in this checkout: track borders and night GT come from our extraction. The
  day reference is the organizers' racing line (data/23-00-38_cut_120m_update/map.mcap, read with their
  tools/track_frame.py), as in his reference.py.
* the mp4 is written with OpenCV (no ffmpeg here); frames are drawn by his Dashboard.render.
"""
import argparse
import json
import os
import sys
import time
import types

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, '.')
from odo import data

PRAF = os.path.abspath('odosseus_prafful')
sys.path.insert(0, os.path.join(PRAF, 'localization'))
from camloc import metrics as cl_metrics  # noqa: E402  (official scoring math, pure numpy)


# ------------------------------------------------------------------ stand-in for his reference.py
def _cut(gt, t0, t1):
    m = (gt['timestamp_ns'] >= t0) & (gt['timestamp_ns'] <= t1)
    return {k: v[m] for k, v in gt.items()}


def _interp_yaw(gt, ts):
    o = np.argsort(gt['timestamp_ns'])
    return np.interp(ts, gt['timestamp_ns'][o], np.unwrap(gt['yaw'][o]))


def _gate():
    a = data.LAPS['L3'][0]                       # start-line crossing = lap start
    x, y, _ = data.interp_gt([data.t0_ns() + int(a * 1e9)])
    return float(x[0]), float(y[0])


ref = types.ModuleType('reference')
ref.GATE, ref.cut, ref.interp_yaw, ref.metrics = _gate(), _cut, _interp_yaw, cl_metrics
sys.modules['reference'] = ref
sys.path.insert(0, os.path.join(PRAF, 'visualization'))
import dashboard as D  # noqa: E402
from live_metrics import RunningMetrics  # noqa: E402
import logging  # noqa: E402
D.mpl.rcParams['font.family'] = ['DejaVu Sans']        # Arial / Helvetica are not installed here (no findfont spam)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


def bounds():
    b = json.load(open(f'{data.DATA}/track_bounds.json'))
    return {k: np.asarray(v, float) for k, v in b.items()}


def gt_dict():
    g = data.gt()
    return {'timestamp_ns': g.timestamp_ns.values.astype(np.int64), 'x': g.x.values, 'y': g.y.values,
            'z': g.z.values, 'yaw': g.yaw.values}


class Line:
    """A reference line with xy_to_sn like the organizers' track_frame.Line: s along, n signed (+ = left)."""

    def __init__(self, x, y, step=0.25):
        s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        g = np.arange(0, s[-1], step)
        self.p = np.c_[np.interp(g, s, x), np.interp(g, s, y)]
        self.s = g
        d = np.gradient(self.p, axis=0)
        self.t = d / np.linalg.norm(d, axis=1, keepdims=True)
        self.tree = cKDTree(self.p)

    def xy_to_sn(self, x, y):
        q = np.c_[np.atleast_1d(x), np.atleast_1d(y)]
        _, i = self.tree.query(q)
        v = q - self.p[i]
        n = self.t[i, 0] * v[:, 1] - self.t[i, 1] * v[:, 0]
        return self.s[i], n, None, None


UPDATE = 'data/23-00-38_cut_120m_update'           # organizers' update: map.mcap (racing line) + tools/track_frame.py


def racing_line():
    """The organizers' racing line as their own track_frame.Line (xy_to_sn: s along, n + = left), as in his reference.py."""
    sys.path.insert(0, os.path.join(UPDATE, 'tools'))
    from track_frame import load_lines
    return load_lines(os.path.join(UPDATE, 'map.mcap'))['track']


def lap_line(lap):
    g = data.gt()
    a, b = data.LAPS[lap]
    g = g[(g.timestamp_ns >= data.t0_ns() + int(a * 1e9)) & (g.timestamp_ns < data.t0_ns() + int(b * 1e9))]
    return g.x.values, g.y.values


# ------------------------------------------------------------------ our run -> his per-frame arrays
FIXED, GATED = ('fix', 'reloc', 'reset'), ('gated', 'reloc_rejected')


def _split(v, n):
    if isinstance(v, str) and v:
        return [int(x) for x in v.split('/')]
    if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v) and n == 1:
        return [int(v)]
    return [0] * n


def thumbs(img_root, ts, cam='fr'):
    f = pd.read_csv(f'{img_root}/images/{cam}.csv').sort_values('stamp_ns')
    st = f.stamp_ns.values
    out, cache = [], {}
    for t in ts:
        i = int(np.clip(np.searchsorted(st, t), 1, len(st) - 1))
        i = i if abs(st[i] - t) < abs(st[i - 1] - t) else i - 1
        p = f'{img_root}/{f.file.values[i]}'
        if p not in cache:
            im = cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_4)
            cache = {p: None if im is None else im[:, :, ::-1].copy()}
        out.append(cache[p])
    return out


def load_run(run, t0=None, t1=None, state=None, img_root=None):
    p = pd.read_csv(f'{run}/pred.csv').sort_values('timestamp_ns').reset_index(drop=True)
    inf = pd.read_csv(f'{run}/online_info.csv')
    cams = json.load(open(f'{run}/config.json'))['cams']
    ts = p.timestamp_ns.values.astype(np.int64)
    m = np.ones(len(p), bool)
    if t0 is not None:
        m &= ts >= ts[0] + int(t0 * 1e9)
    if t1 is not None:
        m &= ts < ts[0] + int(t1 * 1e9)
    p, inf, ts = p[m].reset_index(drop=True), inf[m].reset_index(drop=True), ts[m]
    n, nc = len(p), len(cams)
    st = inf.status.fillna('none').values
    inl = np.array([_split(v, nc) for v in inf.get('inl_cam', pd.Series([''] * n)).values])
    m1 = np.array([_split(v, nc) for v in inf.get('m1_cam', pd.Series([''] * n)).values])
    acc, gat = np.isin(st, FIXED), np.isin(st, GATED)
    used = inl > 0
    cam_state = np.where(used & acc[:, None], 2, np.where(used & gat[:, None], 1, 0))
    cam_text = [[(f'{inl[k, j]} inliers' + (' · accepted' if acc[k] else ' · gated') if used[k, j] else
                  ('no map near' if str(inf.get('map_dist', pd.Series([0] * n)).values[k]) == 'nan' and st[k] == 'none'
                   else 'no fix')) for j in range(nc)] for k in range(n)]
    sx, sy = inf.sx.values, inf.sy.values
    syaw = inf.syaw.values if 'syaw' in inf else np.full(n, np.nan)
    d = {'stamp_ns': ts, 'x': p.x.values, 'y': p.y.values, 'yaw': p.yaw.values,
         'sigma_xy': np.hypot(sx, sy), 'sigma_yaw': syaw,
         'n_fix': (used & (acc | gat)[:, None]).sum(1), 'n_acc': (used & acc[:, None]).sum(1),
         'proc_ms': inf.ms.values.astype(float), 'late_ms': np.zeros(n),
         'speed_kmh': 3.6 * np.interp(ts, state.timestamp_ns.values, state.vx_mps.values),
         'cams': cams, 'cam_matches': m1, 'cam_state': cam_state, 'cam_text': cam_text,
         'status': st, 'reloc_try': inf.reloc_try.fillna(0).values if 'reloc_try' in inf else np.zeros(n),
         'n_inl': inf.n_inl.fillna(0).values, 'bounds': bounds()}
    d['t'] = (ts - ts[0]) / 1e9
    d['thumbs'] = thumbs(img_root, ts) if img_root else None
    return d


# ------------------------------------------------------------------ his dashboards, relabelled for a joint fix
def _fixes(self, ax):
    nc = len(self.d['cams'])
    self.fix_wash = self._dyn(ax.add_patch(D.Polygon(np.zeros((3, 2)), closed=True, fc=D.to_rgba(D.ACC, 0.12), ec='none')))
    self.fix_acc = self._dyn(ax.plot([], [], color=D.ACC, lw=2.0)[0])
    self.fix_rej = self._dyn(ax.plot([], [], color=D.REJ, lw=2.0)[0])
    ax.set_ylim(0, nc + 0.6)
    ax.set_yticks(range(nc + 1))
    D._title(ax, 'Cameras in the joint PnP fix, per frame', f'of {nc} cameras',
             ([D.Line2D([], [], color=D.ACC, lw=2), D.Line2D([], [], color=D.REJ, lw=2)],
              ['fix accepted by the filter', 'fix rejected at the gate']))


class Night(D.TruthDashboard):
    METRIC_ROWS = D.TruthDashboard.METRIC_ROWS[:-1] + [('Frames with an accepted fix', '')]
    _fixes = _fixes


class Day(D.HealthDashboard):
    METRIC_ROWS = [('Frames with an accepted fix', ''), ('Joint PnP fixes', 'accepted · produced · frames'),
                   ('Map matches per camera', 'median, stage 1'), ('Filter position σ', 'now · max'),
                   ('Global retrieval (MegaLoc)', 'tries · accepted'), ('Runtime per frame', '5 %'),
                   ('Coverage', 'ranked ≥ 90 %'), (None, None), ('Offset from the racing line', 'RMS, context only'),
                   ('Ground truth', ''), ('Start pose', '')]
    _fixes = _fixes
    # overview / chase / offset panels: his own (the organizers' racing line, track-frame n)

    def _middle(self, ax):
        D._style(ax, grid=False)
        ax.grid(True, axis='x', color=D.GRID, lw=0.6)
        cams = self.d['cams']
        y = np.arange(len(cams))[::-1].astype(float)
        ax.set_yticks(y, cams)
        ax.set_ylim(-0.7, len(cams) - 0.3)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100], ['0', '25', '50', '75', '≥ 100'])
        self.cam_bars = [self._dyn(ax.add_patch(D.Rectangle((0, yi - 0.28), 0, 0.56, fc=D.MUTED, ec='none', zorder=2))) for yi in y]
        self.cam_lbls = [self._dyn(ax.text(2, yi, '', color=D.INK2, fontsize=8.5, va='center', zorder=4)) for yi in y]
        ax.set_xlabel('guided matches to the night map (joint fix needs ≥ 30 inliers)')
        D._title(ax, 'Camera matching, this frame', 'bars: grey not in a fix · aqua accepted · orange gated', below=True)

    def _metric_values(self, k):
        d, s = self.d, slice(0, k + 1)
        acc = np.isin(d['status'][s], FIXED)
        prod = np.isin(d['status'][s], FIXED + GATED)
        proc = d['proc_ms'][s]
        return [f'{100 * acc.mean():.0f} %', f'{int(acc.sum())} · {int(prod.sum())} · {k + 1}',
                f'{np.median(d["cam_matches"][s]):.0f}',
                f'{d["sigma_xy"][k]:.2f} m · {d["sigma_xy"][s].max():.2f} m',
                f'{int((d["reloc_try"][s] > 0).sum())} · {int(np.isin(d["status"][s], ("reloc", "reset")).sum())}',
                f'{proc.mean():.0f} ms mean · {np.percentile(proc, 95):.0f} ms p95', '100 %',
                f'{np.sqrt(np.mean(d["ref_off"][s] ** 2)):.2f} m', 'none in this bag', d['start_note']]


# ------------------------------------------------------------------ output
def figure():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(16, 9), dpi=120)
    FigureCanvasAgg(fig)
    return fig


def mp4(dash, path, fps=10, hold_s=3.0):
    canvas = dash.fig.canvas
    canvas.draw()
    w, h = canvas.get_width_height()
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    t = time.perf_counter()
    for k in range(dash.n):
        dash.render(k)
        vw.write(cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR))
        if k % 200 == 0:
            print('frame', k, dash.n, flush=True)
    dash.render(dash.n - 1, final=True)
    last = cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    for _ in range(int(hold_s * fps)):
        vw.write(last)
    vw.release()
    return {'frames': dash.n, 'render_s': time.perf_counter() - t, 'size': [w, h]}


def night(a):
    a.fold = a.fold or 'F5'
    d = load_run(a.run, a.t0, a.t1, data.allowed_state(), a.query or data.DATA)
    gt = gt_dict()
    rm = RunningMetrics(gt, d['stamp_ns'], d['x'], d['y'], d['yaw'], d['proc_ms'], d['n_acc'])
    d.update(gt_x=rm.e['gt_x'], gt_y=rm.e['gt_y'], gt_yaw=rm.gt_yaw, lat=rm.e['lat'], lon=rm.e['lon'],
             dyaw_deg=rm.e['dyaw_deg'], ok=rm.ok, running=[rm.at(k) for k in range(len(d['stamp_ns']))])
    d['ref_xy'] = (rm.gt['x'], rm.gt['y'])
    chunk = rm.final()
    cfg = json.load(open(f'{a.run}/config.json'))
    title = f'Night lap {a.lap}' + (f', {a.t0:g}–{a.t1:g} s' if a.t1 else ', whole lap') + ' — final method (Exp24)'
    sub = (f'{cfg.get("split")} not in the map ({a.fold}: night laps {" ".join(data.FOLDS[a.fold]["map_laps"])})  ·  '
           f'{len(d["cams"])} cameras, joint PnP to the night map + car-state EKF  ·  10 Hz image sets')
    return d, Night, title, sub, {'chunk': chunk}


def day(a):
    q = data.Run(a.query)
    d = load_run(a.run, a.t0, a.t1, q.allowed_state(), a.query)
    line = racing_line()
    s, n, _, _ = line.xy_to_sn(d['x'], d['y'], t_ns=d['stamp_ns'])
    d.update(ref_s=s, ref_off=n, ref_xy=(line.p[:, 0], line.p[:, 1]), dt_end=float(np.round(d['t'][-1] + 0.1)))
    info = json.load(open(f'{a.query}/first_pose_standin.json')) if os.path.exists(f'{a.query}/first_pose_standin.json') else {}
    cfg = json.load(open(f'{a.run}/config.json'))['cfg']
    d['start_note'] = f'stand-in, σ {cfg.get("init_sigma_xy", 0.05):g} m / {cfg.get("init_sigma_yaw_deg", 0.11):g}°'
    title = 'L9: daytime lap — final method (Exp24)' + (f', {a.t0:g}–{a.t1:g} s' if a.t1 else '')
    maps = (f'map = {a.fold}: night laps {" ".join(data.FOLDS[a.fold]["map_laps"])}' if a.fold
            else 'map = all night laps L0–L8')
    if a.fold:
        title = title.replace('(Exp24)', f'(Exp24), {a.fold} map')
    sub = (f'No ground truth: health signals only  ·  {maps}  ·  {len(d["cams"])} cameras, '
           'joint PnP to the night map + car-state EKF  ·  stand-in first pose')
    acc = np.isin(d['status'], FIXED)
    return d, Day, title, sub, {'start_pose': {k: v for k, v in info.items() if k != 'dr_track'},
                                'fix_frames_pct': float(100 * acc.mean()),
                                'gated_pct': float(100 * np.isin(d['status'], GATED).mean()),
                                'sigma_xy_median_m': float(np.median(d['sigma_xy'])),
                                'sigma_xy_max_m': float(d['sigma_xy'].max()),
                                'racing_line_offset_m': dict(rms=float(np.sqrt(np.mean(n ** 2))),
                                                             p95_abs=float(np.percentile(np.abs(n), 95)),
                                                             max_abs=float(np.abs(n).max())),
                                'cam_matches_median': np.median(d['cam_matches'], 0).tolist()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('night', 'day')); ap.add_argument('run')
    ap.add_argument('--query', help='image root (day: the extracted L9 bag)')
    ap.add_argument('--lap', default='L2')
    ap.add_argument('--fold', help='map fold of the run (night: default F5; day: default = all night laps)')
    ap.add_argument('--t0', type=float); ap.add_argument('--t1', type=float); ap.add_argument('--name')
    a = ap.parse_args()
    d, Dash, title, sub, extra = (night if a.mode == 'night' else day)(a)
    name = a.name or (f'{a.mode}' + (f'_{a.t0:g}-{a.t1:g}s' if a.t1 else '_full'))
    out = f'{a.run}/prafful_viz/{name}'
    os.makedirs(out, exist_ok=True)
    dash = Dash(d, figure(), title, sub)
    res = mp4(dash, f'{out}/{name}.mp4')
    dash.fig.savefig(f'{out}/final_frame.png', facecolor=dash.fig.get_facecolor())
    summary = dict(run=a.run, mode=a.mode, frames=len(d['t']), cams=d['cams'], mp4=res,
                   runtime_ms=dict(mean=float(d['proc_ms'].mean()), p95=float(np.percentile(d['proc_ms'], 95))),
                   speed_kmh=[float(d['speed_kmh'].min()), float(d['speed_kmh'].max())], **extra)
    json.dump(summary, open(f'{out}/metrics.json', 'w'), indent=1, default=float)
    print('VIZ_DONE', out, json.dumps({k: v for k, v in summary.items() if k not in ('mp4',)}, default=float)[:600], flush=True)
