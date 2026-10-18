"""A run of the localiser (odo.online: pred.csv, online_info.csv, config.json) as the dashboard's per-frame arrays,
and the two dashboards relabelled for its measurement: ONE joint PnP fix over all cameras per image set.

* "Cameras in the joint fix" = cameras with PnP inliers in that fix, accepted by the filter or rejected at the gate.
* camera bars = stage-1 guided matches of each camera to the map (the joint fix needs >= 30 inliers in total).
* relocalisation (MegaLoc) shows up as status reloc / reset in the fixes and in the metrics.
"""
import json
import os

import cv2
import numpy as np
import pandas as pd

import dashboard as D
from reference import data

ACCEPTED, GATED = ('fix', 'reloc', 'reset'), ('gated', 'reloc_rejected')


def _split(v, n):
    """'12/0/40/...' (per camera, online_info.csv) -> [12, 0, 40, ...]; anything else -> zeros."""
    if isinstance(v, str) and v:
        return [int(x) for x in v.split('/')]
    return [0] * n


def thumbs(frames, ts):
    """Camera image nearest to each stamp, at 1/4 resolution (RGB). frames: stamp_ns, path (data.frames / Run.frames)."""
    st, paths = frames.stamp_ns.values, frames.path.values
    out, cache = [], {}
    for t in ts:
        i = int(np.clip(np.searchsorted(st, t), 1, len(st) - 1))
        i = i if abs(st[i] - t) < abs(st[i - 1] - t) else i - 1
        p = paths[i]
        if p not in cache:
            im = cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_4)
            cache = {p: None if im is None else im[:, :, ::-1].copy()}
        out.append(cache[p])
    return out


def running_coverage(t):
    """Share of the 1 s intervals begun so far that have a pose, per frame [%]."""
    b = np.floor(t).astype(int)
    seen, cov = set(), np.zeros(len(t))
    for k, v in enumerate(b):
        seen.add(v)
        cov[k] = 100.0 * len(seen) / (v + 1)
    return cov


def load_run(run, state, frames=None, t0=None, t1=None):
    """Per-frame arrays of an odo.online run. state: allowed car state (timestamp_ns, v_mps); frames: the thumbnail
    camera's frame table, or None for no thumbnail; t0 / t1: chunk in s after the first pose."""
    p = pd.read_csv(f'{run}/pred.csv')
    inf = pd.read_csv(f'{run}/online_info.csv')
    cfg = json.load(open(f'{run}/config.json'))
    cams = cfg['cams']
    ts = p.timestamp_ns.values.astype(np.int64)
    m = np.ones(len(p), bool)
    if t0 is not None:
        m &= ts >= ts[0] + int(t0 * 1e9)
    if t1 is not None:
        m &= ts < ts[0] + int(t1 * 1e9)
    p, inf, ts = p[m].reset_index(drop=True), inf[m].reset_index(drop=True), ts[m]
    n, nc = len(p), len(cams)
    col = lambda c, fill: inf[c].values if c in inf else np.full(n, fill)
    st = inf.status.fillna('none').values
    inl = np.array([_split(v, nc) for v in col('inl_cam', '')])
    m1 = np.array([_split(v, nc) for v in col('m1_cam', '')])
    acc, gat = np.isin(st, ACCEPTED), np.isin(st, GATED)
    used = inl > 0
    nomap = pd.Series(col('map_dist', np.nan)).isna().values & (st == 'none')
    cam_state = np.where(used & acc[:, None], 2, np.where(used & gat[:, None], 1, 0))
    cam_text = [[(f'{inl[k, j]} inliers · ' + ('accepted' if acc[k] else 'gated')) if used[k, j]
                 else ('no map near' if nomap[k] else 'no fix') for j in range(nc)] for k in range(n)]
    t = (ts - ts[0]) / 1e9
    d = {'stamp_ns': ts, 't': t, 'x': p.x.values, 'y': p.y.values, 'yaw': p.yaw.values,
         'sigma_xy': np.hypot(inf.sx.values, inf.sy.values), 'sigma_yaw': col('syaw', np.nan).astype(float),
         'n_fix': (used & (acc | gat)[:, None]).sum(1), 'n_acc': (used & acc[:, None]).sum(1),
         'proc_ms': inf.ms.values.astype(float),
         'speed_kmh': 3.6 * np.interp(ts, state.timestamp_ns.values, state.v_mps.values),
         'cams': cams, 'cam_matches': m1, 'cam_state': cam_state, 'cam_text': cam_text, 'status': st,
         'reloc_try': np.nan_to_num(col('reloc_try', 0).astype(float)), 'coverage': running_coverage(t),
         'cfg': cfg}
    d['thumbs'] = thumbs(frames, ts) if frames is not None else None
    return d


def _fixes(self, ax):
    """Fixes panel for a joint fix: how many cameras contributed inliers, accepted vs rejected at the gate."""
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
    """Mapped mode: a lap with ground truth (errors, official metrics)."""
    _fixes = _fixes

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.ax_fix.set_title(f'Cameras in the joint PnP fix (of {len(self.d["cams"])})', loc='left',
                              color=D.INK, fontsize=10.5, pad=17)


class Day(D.HealthDashboard):
    """Test mode: a run without ground truth (health signals; the organizers' racing line as context)."""
    METRIC_ROWS = [('Frames with an accepted fix', ''), ('Joint PnP fixes', 'accepted · produced · frames'),
                   ('Map matches per camera', 'median, stage 1'), ('Filter position σ', 'now · max'),
                   ('Relocalisation (MegaLoc)', 'tries · jumps accepted'), ('Runtime per frame', '5 %'),
                   ('Coverage', 'ranked ≥ 90 %'), (None, None), ('Offset from the racing line', 'RMS, context only'),
                   ('Ground truth', ''), ('Start pose', '')]
    _fixes = _fixes

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
        ax.set_xlabel('guided matches to the map (the joint fix needs ≥ 30 inliers)')
        D._title(ax, 'Camera matching, this frame', 'bars: grey not in a fix · aqua accepted · orange gated', below=True)

    def _metric_values(self, k):
        d, s = self.d, slice(0, k + 1)
        acc = np.isin(d['status'][s], ACCEPTED)
        prod = np.isin(d['status'][s], ACCEPTED + GATED)
        proc = d['proc_ms'][s]
        off = d['ref_off'][s]
        return [f'{100 * acc.mean():.0f} %', f'{int(acc.sum())} · {int(prod.sum())} · {k + 1}',
                f'{np.median(d["cam_matches"][s]):.0f}',
                f'{d["sigma_xy"][k]:.2f} m · {d["sigma_xy"][s].max():.2f} m',
                f'{int((d["reloc_try"][s] > 0).sum())} · {int(np.isin(d["status"][s], ("reloc", "reset")).sum())}',
                f'{proc.mean():.0f} ms mean · {np.percentile(proc, 95):.0f} ms p95', f'{d["coverage"][k]:.0f} %',
                f'{np.sqrt(np.mean(off ** 2)):.2f} m' if d['has_ref'] else 'no racing line', 'none in this run',
                d['start_note']]
