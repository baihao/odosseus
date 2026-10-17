"""Evaluate a camera VO trajectory: convert to map poses (speed-scaled, anchored at first pose),
write pred CSV + PDF/HTML/video reports.

    python -m odo.eval_vo --poses <file> --fmt c2w|mast3r --seq data/prepared/r_val --out outputs/<exp> \
        --name "..." [--split val] [--video]

c2w    : one row per frame, 16 numbers (4x4 camera-to-world), e.g. VGGT-Long camera_poses.txt
mast3r : frames_T_WC.txt from the patched MASt3R-SLAM (idx kfflag t q s); kfflag -1 = tracking lost
tum    : stamp tx ty tz qx qy qz qw (one row per frame, same order as stamps.csv)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from . import report, vo2map


def load_poses(path, fmt, n):
    valid = np.ones(n, bool)
    if fmt == 'c2w':
        T = np.loadtxt(path).reshape(-1, 4, 4)
    elif fmt == 'mast3r':
        a = np.loadtxt(path)
        idx = a[:, 0].astype(int)
        T = np.tile(np.eye(4), (n, 1, 1)); valid[:] = False
        T[idx] = vo2map.sim3_rows_to_T(a[:, 2:10]); valid[idx] = a[:, 1] >= 0
        valid[0] = True
    elif fmt == 'tum':
        a = np.loadtxt(path)
        T = vo2map.sim3_rows_to_T(np.concatenate([a[:, 1:8], np.ones((len(a), 1))], 1))
    else:
        raise ValueError(fmt)
    # orthonormalise rotations (chunk alignment may leave tiny scale residue)
    for i in range(len(T)):
        U, _, Vt = np.linalg.svd(T[i, :3, :3]); T[i, :3, :3] = U @ Vt
    return T[:n], valid[:len(T)]


def run(poses, fmt, seq, out, name, split='val', video=False, extra=None, others=None, baselines=None):
    os.makedirs(out, exist_ok=True)
    st = pd.read_csv(f'{seq}/stamps.csv')
    cam = json.load(open(f'{seq}/intrinsics.json'))['cam']
    ts = st.stamp_ns.values
    T, valid = load_poses(poses, fmt, len(ts))
    ts = ts[:len(T)]
    res = {}
    for mode in ['heading', 'speed', 'vo']:
        k = vo2map.fit_vo_scale(ts, T, cam) if mode == 'vo' else 1.0
        x, y, yaw = vo2map.to_map(ts, T, cam, scale=mode, valid=valid, vo_scale=k)
        p = f'{out}/pred_{mode}.csv'
        pd.DataFrame(dict(timestamp_ns=ts, x=x, y=y, yaw=yaw)).to_csv(p, index=False, float_format='%.5f')
        ex = dict(extra or {}, scale_mode=mode, camera=cam, valid_frames_pct=float(100 * valid.mean()))
        if mode == 'vo':
            ex['oracle_global_scale'] = k
        sub = f'{out}/{mode}'
        os.makedirs(sub, exist_ok=True)
        os.replace(p, f'{sub}/pred.csv')
        res[mode] = report.make_report(f'{sub}/pred.csv', f'{name} [{mode} scale]', split,
                                       video=video and mode == 'heading', extra=ex,
                                       others=others, baselines=baselines)
    json.dump(res, open(f'{out}/summary.json', 'w'), indent=1)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--poses'); ap.add_argument('--fmt', default='c2w'); ap.add_argument('--seq')
    ap.add_argument('--out'); ap.add_argument('--name', default='VO'); ap.add_argument('--split', default='val')
    ap.add_argument('--video', action='store_true')
    a = ap.parse_args()
    bl = {'dead-reckoning': f'{report.data.ROOT}/outputs/01_deadreckoning/metrics_val.json'}
    oth = {'dead-reckoning': f'{report.data.ROOT}/outputs/01_deadreckoning/pred_val.csv'}
    r = run(a.poses, a.fmt, a.seq, a.out, a.name, a.split, a.video, baselines=bl, others=oth)
    for k, m in r.items():
        print(k, {kk: round(v, 3) for kk, v in m.items() if isinstance(v, float)})
