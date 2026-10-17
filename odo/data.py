"""Dataset access: GT, car state, calibration, camera frame indices, splits."""
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'extracted')
CAMS = ['fl', 'fr', 'cl', 'cr', 'r', 'rl', 'rr']

# Lap boundaries (start-line crossings, s since first GT stamp) found from the
# sn idx wrap in state_estimation, see outputs/00_data_overview/timeline.png.
# Validation = the last two full laps (fastest ones); never tune on them.
SPLITS = {
    'train': (0.0, 897.0),
    'val': (897.0, 1120.0),
    'tune': (632.0, 752.0),   # one training lap used for tuning VO/fusion parameters
    'dev2': (752.0, 897.0),   # second development lap (stage 4), map = train minus this lap +-10 s
    'all': (0.0, 1e9),
}


@lru_cache(None)
def gt():
    d = pd.read_csv(f'{DATA}/gt.csv')
    d = d.sort_values('timestamp_ns').drop_duplicates('timestamp_ns').reset_index(drop=True)
    return d


def t0_ns():
    return int(gt().timestamp_ns.iloc[0])


@lru_cache(None)
def state():
    """Car state. Only speed / accelerations / yaw rate are allowed as method inputs."""
    d = pd.read_csv(f'{DATA}/state.csv')
    d['timestamp_ns'] = d['header.stamp.sec'].astype(np.int64) * 10**9 + d['header.stamp.nanosec'].astype(np.int64)
    return d.sort_values('timestamp_ns').reset_index(drop=True)


ALLOWED_STATE = ['timestamp_ns', 'v_mps', 'vx_mps', 'vy_mps', 'wz_radps', 'ax_mps2', 'ay_mps2']


def allowed_state():
    return state()[ALLOWED_STATE].copy()


@lru_cache(None)
def calib():
    c = json.load(open(f'{DATA}/calib.json'))
    for v in c.values():
        v['K'] = np.array(v['K']).reshape(3, 3)
        v['D'] = np.array(v['D'])
        v['T_base_cam'] = np.array(v['T_base_cam'])
    return c


@lru_cache(None)
def frames(cam):
    d = pd.read_csv(f'{DATA}/images/{cam}.csv')
    d['path'] = DATA + '/' + d.file
    return d.sort_values('stamp_ns').reset_index(drop=True)


def frames_split(cam, split='val', stride=1):
    a, b = SPLITS[split]
    f = frames(cam)
    t = (f.stamp_ns - t0_ns()) / 1e9
    return f[(t >= a) & (t < b)].iloc[::stride].reset_index(drop=True)


@lru_cache(None)
def track_bounds():
    return {k: np.array(v) for k, v in json.load(open(f'{DATA}/track_bounds.json')).items()}


def interp_gt(ts_ns):
    """GT x,y,yaw (yaw unwrapped then rewrapped) linearly interpolated at ts_ns."""
    g = gt()
    tg = g.timestamp_ns.values.astype(np.float64)
    ts = np.asarray(ts_ns, dtype=np.float64)
    x = np.interp(ts, tg, g.x.values)
    y = np.interp(ts, tg, g.y.values)
    yaw = np.interp(ts, tg, np.unwrap(g.yaw.values))
    return x, y, np.angle(np.exp(1j * yaw))


def interp_state(ts_ns, cols=('v_mps', 'wz_radps')):
    s = state()
    tg = s.timestamp_ns.values.astype(np.float64)
    return {c: np.interp(np.asarray(ts_ns, dtype=np.float64), tg, s[c].values) for c in cols}


def first_pose(ts_ns):
    """Initial pose allowed by the rules: GT at the first stamp of the run/segment."""
    x, y, yaw = interp_gt([ts_ns])
    return float(x[0]), float(y[0]), float(yaw[0])


class Run:
    """Query-side access to an extracted run (e.g. the hidden test run): images, allowed car-state fields,
    calibration, and ONLY the first pose of the run (rules: the first pose may be used as the initial pose)."""

    def __init__(self, root):
        self.root = root
        self._f = {}

    def frames(self, cam):
        if cam not in self._f:
            d = pd.read_csv(f'{self.root}/images/{cam}.csv')
            d['path'] = self.root + '/' + d.file
            self._f[cam] = d.sort_values('stamp_ns').reset_index(drop=True)
        return self._f[cam]

    def allowed_state(self):
        d = pd.read_csv(f'{self.root}/state.csv', usecols=lambda c: c in ('header.stamp.sec', 'header.stamp.nanosec')
                        or c in ALLOWED_STATE)
        d['timestamp_ns'] = d['header.stamp.sec'].astype(np.int64) * 10**9 + d['header.stamp.nanosec'].astype(np.int64)
        return d.sort_values('timestamp_ns').reset_index(drop=True)[ALLOWED_STATE]

    def calib(self):
        c = json.load(open(f'{self.root}/calib.json'))
        for v in c.values():
            v['K'] = np.array(v['K']).reshape(3, 3); v['D'] = np.array(v['D']); v['T_base_cam'] = np.array(v['T_base_cam'])
        return c

    def first_pose(self):
        """(stamp_ns, x, y, yaw) of the first pose of the run: first_pose.csv if given, else the first GT row."""
        p = f'{self.root}/first_pose.csv'
        r = pd.read_csv(p if os.path.exists(p) else f'{self.root}/gt.csv', nrows=1).iloc[0]
        return int(r.timestamp_ns), float(r.x), float(r.y), float(r.yaw)
