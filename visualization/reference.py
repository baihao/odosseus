"""Reference data for the visualizer: ground truth (scoring and display only, never an input of the localiser),
track borders, the start/finish line, the organizers' racing line, and the official scoring math.

Everything comes from this repository's odo.data (data/extracted, or the checkpoints' train_ref.npz) and the
dataset's own files; odo.online is not changed by the visualizer.
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from odo import data  # noqa: E402


def _first(*paths):
    return next((p for p in paths if os.path.exists(p)), paths[0])


DATASET = _first(os.path.join(ROOT, 'data', 'c0_2026-10-13_22-53-42_camloc'),
                 os.path.join(ROOT, 'hackFiles', 'c0_2026-10-13_22-53-42_camloc'))
OFFICIAL_EVAL = os.path.join(DATASET, 'tools', 'evaluate.py')
UPDATE = _first(os.path.join(ROOT, 'data', '23-00-38_cut_120m_update'),          # organizers' track-frame update
                os.path.join(ROOT, 'hackFiles', '23-00-38_cut_120m_update'))


class metrics:
    """The official scoring math (tools/evaluate.py; = odo.metrics.challenge_metrics) on a given GT cut."""
    BIN_NS = 1_000_000_000
    MAX_GT_GAP_NS = 50_000_000

    @staticmethod
    def errors(gt, ts, x, y, yaw):
        """Per pose: GT interpolated at the stamp; position, lateral / longitudinal (GT car frame) and heading
        error; ok = GT on both sides within 50 ms (the poses evaluate.py scores)."""
        o = np.argsort(gt['timestamp_ns'])
        tg = gt['timestamp_ns'][o]
        ts = np.asarray(ts, np.int64)
        i = np.searchsorted(tg, ts)
        ok = (i > 0) & (i < len(tg))
        i = np.clip(i, 1, len(tg) - 1)
        ok &= (tg[i] - tg[i - 1]) <= metrics.MAX_GT_GAP_NS
        a = (ts - tg[i - 1]) / (tg[i] - tg[i - 1])
        gx_, gy_, gyaw_ = gt['x'][o], gt['y'][o], np.unwrap(gt['yaw'][o])
        gx = gx_[i - 1] + a * (gx_[i] - gx_[i - 1])
        gy = gy_[i - 1] + a * (gy_[i] - gy_[i - 1])
        gyaw = gyaw_[i - 1] + a * (gyaw_[i] - gyaw_[i - 1])
        dx, dy = np.asarray(x) - gx, np.asarray(y) - gy
        return {'ok': ok, 'gt_x': gx, 'gt_y': gy, 'pos': np.hypot(dx, dy),
                'lon': np.cos(gyaw) * dx + np.sin(gyaw) * dy, 'lat': -np.sin(gyaw) * dx + np.cos(gyaw) * dy,
                'dyaw_deg': np.degrees(np.angle(np.exp(1j * (np.asarray(yaw) - gyaw))))}

    @staticmethod
    def score(gt, ts, x, y, yaw):
        e = metrics.errors(gt, ts, x, y, yaw)
        ok = e['ok']
        pos = e['pos'][ok]
        rms = lambda v: float(np.sqrt(np.mean(v ** 2))) if len(v) else float('nan')
        tg = np.sort(gt['timestamp_ns'])
        bins = np.arange(tg[0], tg[-1], metrics.BIN_NS)
        covered = np.unique(np.searchsorted(bins, np.asarray(ts, np.int64)[ok], side='right') - 1)
        covered = covered[(covered >= 0) & (covered < len(bins))]
        return {'n_submitted': int(len(ts)), 'n_evaluated': int(ok.sum()),
                'coverage_pct': 100.0 * len(covered) / max(len(bins), 1),
                'pos_rmse_m': rms(pos), 'pos_median_m': float(np.median(pos)), 'pos_p95_m': float(np.percentile(pos, 95)),
                'lat_rmse_m': rms(e['lat'][ok]), 'lon_rmse_m': rms(e['lon'][ok]), 'yaw_rmse_deg': rms(e['dyaw_deg'][ok]),
                'within_0.25_pct': 100.0 * float(np.mean(pos < 0.25)), 'within_0.5_pct': 100.0 * float(np.mean(pos < 0.5)),
                'within_1.0_pct': 100.0 * float(np.mean(pos < 1.0))}

    @staticmethod
    def run_official(eval_py, gt_csv, pred_csv):
        """The dataset's own tools/evaluate.py on a chunk (its printed table), if the dataset is present."""
        if not os.path.exists(eval_py):
            return f'(official scorer not found: {eval_py})'
        return subprocess.run([sys.executable, eval_py, gt_csv, pred_csv], capture_output=True, text=True).stdout


def ground_truth():
    """GT of the training run (timestamp_ns, x, y, z, yaw) as arrays."""
    g = data.gt()
    return {'timestamp_ns': g.timestamp_ns.values.astype(np.int64), 'x': g.x.values.astype(float),
            'y': g.y.values.astype(float), 'z': g.z.values.astype(float) if 'z' in g else np.zeros(len(g)),
            'yaw': g.yaw.values.astype(float)}


def _gate():
    """Start/finish line: the GT position at the start of the first flying lap (laps start at the line)."""
    x, y, _ = data.interp_gt([data.t0_ns() + int(data.LAPS['L3'][0] * 1e9)])
    return float(x[0]), float(y[0])


try:
    GATE = _gate()
except FileNotFoundError:          # no training data at all: no gate marker
    GATE = (np.nan, np.nan)


def track_bounds(root=None):
    """{name: (N, 3) points in map}: the track borders of an extracted run (default: the training run)."""
    for p in ([f'{root}/track_bounds.json'] if root else []) + [f'{data.DATA}/track_bounds.json']:
        if os.path.exists(p):
            b = json.load(open(p))
            if b:
                return {k: np.asarray(v, float) for k, v in b.items()}
    return {}


def racing_line():
    """The organizers' racing line (update map.mcap) as their own tools/track_frame.Line: xy_to_sn gives (s, n),
    n = signed distance from the line, + = left. None if the update folder is not present."""
    tools = os.path.join(UPDATE, 'tools')
    if not os.path.exists(os.path.join(UPDATE, 'map.mcap')):
        return None
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from track_frame import load_lines
    return load_lines(os.path.join(UPDATE, 'map.mcap'))['track']


def cut(gt, t0_ns, t1_ns):
    m = (gt['timestamp_ns'] >= t0_ns) & (gt['timestamp_ns'] <= t1_ns)
    return {k: v[m] for k, v in gt.items()}


def interp_yaw(gt, ts):
    """GT yaw at ts, interpolated like metrics.errors (on the unwrapped angle)."""
    o = np.argsort(gt['timestamp_ns'])
    return np.interp(ts, gt['timestamp_ns'][o], np.unwrap(gt['yaw'][o]))


def write_gt_csv(gt, path):
    """GT in the format of the official tools/extract_gt.py, for running tools/evaluate.py on a chunk."""
    with open(path, 'w') as f:
        f.write('timestamp_ns,x,y,z,yaw\n')
        for t, x, y, z, a in zip(gt['timestamp_ns'], gt['x'], gt['y'], gt['z'], gt['yaw']):
            f.write(f'{int(t)},{x:.4f},{y:.4f},{z:.4f},{a:.6f}\n')
