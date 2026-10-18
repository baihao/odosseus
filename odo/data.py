"""Dataset access: GT, car state, calibration, camera frame indices, splits."""
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'extracted')
CAMS = ['fl', 'fr', 'cl', 'cr', 'r', 'rl', 'rr']

# Laps L0..L8 (start-line crossings, s since the first GT stamp).
_LAP_STARTS = [0.0, 185.456, 365.569, 523.375, 633.972, 752.192, 896.116, 1007.673, 1120.703, 1194.393]
LAPS = {f'L{i}': (_LAP_STARTS[i], _LAP_STARTS[i + 1]) for i in range(9)}

# Current protocol: 3-fold cross-validation over laps. L5 is always the tuning lap and never in a map;
# in F1-F3 each of L1, L2, L3, L4, L6, L7 is a validation lap once; F4 adds L0 (out-lap) and re-validates L4, L7 with a
# 5-lap map. map_cv_fN = feature map restricted to the fold's map laps.
TUNE_LAP = 'L5'
FOLDS = {
    'F1': dict(map='map_cv_f1', map_laps=['L0', 'L2', 'L3', 'L4', 'L7', 'L8'], val=['L1', 'L6']),
    'F2': dict(map='map_cv_f2', map_laps=['L0', 'L1', 'L3', 'L4', 'L6', 'L8'], val=['L2', 'L7']),
    'F3': dict(map='map_cv_f3', map_laps=['L0', 'L1', 'L2', 'L6', 'L7', 'L8'], val=['L3', 'L4']),
    'F4': dict(map='map_cv_f4', map_laps=['L1', 'L2', 'L3', 'L6', 'L8'], val=['L0', 'L4', 'L7']),
    # F5: F2's map with the roles of L2 and L5 swapped (L2 = tuning lap of this fold). F6: F1's map with L2 replaced by
    # L5 (L2 = tuning lap). Parameters stay those chosen on L5, so for these folds L5 is not an unseen lap.
    'F5': dict(map='map_cv_f2_swap', map_laps=['L0', 'L1', 'L3', 'L4', 'L6', 'L8'], val=['L5', 'L7'], tune='L2'),
    'F6': dict(map='map_cv_f1_swap', map_laps=['L0', 'L5', 'L3', 'L4', 'L7', 'L8'], val=['L1', 'L6'], tune='L2'),
    # F7: the test-run setting for daylight. Map = all night laps (as for the hidden test run); validation = the day
    # proxy of L2 (night images translated to day), whose night frames are in the map: only the appearance changes.
    'F7': dict(map='map_all', map_laps=[f'L{i}' for i in range(9)], val=['L2'], tune='L5', day_query=True),
}


def tune_lap(fold):
    return FOLDS[fold].get('tune', TUNE_LAP)


def fold_windows(fold):
    """Map time windows (ns) of a fold: its map laps."""
    return [(t0_ns() + int(LAPS[l][0] * 1e9), t0_ns() + int(LAPS[l][1] * 1e9)) for l in FOLDS[fold]['map_laps']]


# Legacy split (stages 1-2 and the first Exp21 numbers), kept for the old experiments:
# train = 0-897 s, tune ~ L4, dev2 ~ L5, val ~ L6 + L7.
SPLITS = {
    'train': (0.0, 897.0),
    'val': (897.0, 1120.0),
    'tune': (632.0, 752.0),   # one training lap used for tuning VO/fusion parameters
    'dev2': (752.0, 897.0),   # second development lap (stage 4), map = train minus this lap +-10 s
    'all': (0.0, 1e9),
    **LAPS,
}


# Inference without the training bag: the training-run columns that test mode reads (map-run GT poses for the
# relocalisation database and the stand-in first pose, the state estimator's pos_accuracy for the map filter),
# shipped with the checkpoints. Written by scripts/build_train_ref.py.
TRAIN_REF = os.path.join(ROOT, 'outputs', '20_map', 'train_ref.npz')


def _train_ref(name):
    if not os.path.exists(TRAIN_REF):
        raise FileNotFoundError(f'training data not found: neither {DATA} (scripts/extract_mcap.py on the training '
                                f'bag) nor {TRAIN_REF} (checkpoints, scripts/download_checkpoints.sh)')
    z = np.load(TRAIN_REF)
    cols = [str(c) for c in z[f'{name}_cols']]
    d = pd.DataFrame({c: z[f'{name}_{i}'] for i, c in enumerate(cols)})
    return d


@lru_cache(None)
def gt():
    d = pd.read_csv(f'{DATA}/gt.csv') if os.path.exists(f'{DATA}/gt.csv') else _train_ref('gt')
    d = d.sort_values('timestamp_ns').drop_duplicates('timestamp_ns').reset_index(drop=True)
    return d


def t0_ns():
    return int(gt().timestamp_ns.iloc[0])


@lru_cache(None)
def state():
    """Car state. Only speed / accelerations / yaw rate are allowed as method inputs (pos_accuracy: map filter only)."""
    if not os.path.exists(f'{DATA}/state.csv'):
        return _train_ref('state')        # timestamp_ns, pos_accuracy; stored in the sorted order of the csv path
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

    def first_pose(self, path=None):
        """(stamp_ns or None, x, y, yaw) of the given first pose of the run (rules: allowed as the initial pose).
        path (odo.online --first_pose FILE), else <root>/first_pose.csv, else the first row of <root>/gt.csv (the
        first /tf of the bag). Only the first row is read. No timestamp_ns column: the pose at the first image."""
        cands = [path] if path else [f'{self.root}/first_pose.csv', f'{self.root}/gt.csv']
        p = next((c for c in cands if os.path.exists(c)), None)
        if p is None:
            raise FileNotFoundError(f'no first pose: {" / ".join(cands)} not found; give --first_pose FILE '
                                    f'(csv: timestamp_ns,x,y,yaw) or --no_first_pose')
        d = pd.read_csv(p, nrows=1)
        if not len(d):
            raise ValueError(f'no first pose: {p} has no rows (no /tf in the bag?); give --first_pose FILE')
        r = d.iloc[0]
        if 'yaw' in r:
            yaw = float(r.yaw)
        elif all(k in r for k in ('qx', 'qy', 'qz', 'qw')):        # heading from the quaternion (map -> base_link)
            yaw = float(np.arctan2(2 * (r.qw * r.qz + r.qx * r.qy), 1 - 2 * (r.qy ** 2 + r.qz ** 2)))
        else:
            raise ValueError(f'first pose {p}: needs columns x, y and yaw (or qx, qy, qz, qw)')
        return (int(r.timestamp_ns) if 'timestamp_ns' in r else None), float(r.x), float(r.y), yaw
