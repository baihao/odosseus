"""L9 (daytime bag, no GT, no first pose): a stand-in first pose for the run, written as <root>/first_pose.csv.

Port of odosseus_prafful/visualization/daybag.stand_in_first_pose: the whole bag's car-state motion (vx, vy, wz
integrated over distance) is matched by curvature over distance against the GT path of a night lap (L7), best shift
by correlation; the heading offset is the median over the matched distance; then dead reckoning from the matched
start, read at the first image. It uses the whole bag (not causal) and only replaces the given first pose of the
real test run. Its error is a few metres, so the run starts with a wide initial sigma (init_sigma_xy).

    python scripts/l9_first_pose.py data/L9/extracted [--lap L7]
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from odo import data


def curvature(yaw, ds, window_m=10.0):
    k = np.gradient(yaw, ds)
    w = max(1, int(window_m / ds))
    return np.convolve(k, np.ones(w) / w, mode='same')


def stand_in(root, lap='L7', ds=0.5):
    q = data.Run(root)
    st = q.allowed_state()
    t_ns = int(min(q.frames(c).stamp_ns.iloc[0] for c in data.CAMS))
    g = data.gt()
    a, b = data.LAPS[lap]
    g = g[(g.timestamp_ns >= data.t0_ns() + int(a * 1e9)) & (g.timestamp_ns < data.t0_ns() + int(b * 1e9))]
    gx, gy, gyaw = g.x.values, g.y.values, np.unwrap(g.yaw.values)
    sr = np.r_[0, np.cumsum(np.hypot(np.diff(gx), np.diff(gy)))]
    grid = np.arange(0, sr[-1], ds)
    rx, ry, ryaw = np.interp(grid, sr, gx), np.interp(grid, sr, gy), np.interp(grid, sr, gyaw)
    ts = st.timestamp_ns.values
    t = (ts - ts[0]) / 1e9
    dt = np.diff(t, prepend=t[0])
    vx, vy, wz = st.vx_mps.values, st.vy_mps.values, st.wz_radps.values
    sd, yd = np.cumsum(vx * dt), np.cumsum(wz * dt)
    yds = np.interp(np.arange(0, sd[-1], ds), sd, yd)
    kr, kd = curvature(ryaw, ds), curvature(yds, ds)
    n = min(len(kd), len(kr))
    kd, yds = kd[:n] - kd[:n].mean(), yds[:n]
    kr2 = np.r_[kr, kr]                                   # closed lap: the reference wraps around
    score = np.array([np.dot(kr2[i:i + n] - kr2[i:i + n].mean(), kd) for i in range(len(kr))])
    sh = int(np.argmax(score))
    ryaw2 = np.r_[ryaw, ryaw + (ryaw[-1] - ryaw[0])]
    off = float(np.median(ryaw2[sh:sh + n] - yds))
    psi = off + yd
    x = rx[sh] + np.cumsum((vx * np.cos(psi) - vy * np.sin(psi)) * dt)
    y = ry[sh] + np.cumsum((vx * np.sin(psi) + vy * np.cos(psi)) * dt)
    tq = (t_ns - ts[0]) / 1e9
    if tq <= 0:                                           # image before the first car-state sample: extrapolate
        x0 = x[0] + (vx[0] * np.cos(psi[0]) - vy[0] * np.sin(psi[0])) * tq
        y0 = y[0] + (vx[0] * np.sin(psi[0]) + vy[0] * np.cos(psi[0])) * tq
        a0 = psi[0] + wz[0] * tq
    else:
        x0, y0, a0 = np.interp(tq, t, x), np.interp(tq, t, y), np.interp(tq, t, psi)
    a0 = float(np.arctan2(np.sin(a0), np.cos(a0)))
    info = dict(method=f'curvature over distance matched against {lap} GT (whole bag, not causal)', ref_lap=lap,
                start_along_ref_m=sh * ds, heading_offset_deg=float(np.degrees(off)), stamp_ns=t_ns,
                pose=[float(x0), float(y0), a0], dr_track=dict(x=x[::10].tolist(), y=y[::10].tolist()))
    return (t_ns, float(x0), float(y0), a0), info


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root'); ap.add_argument('--lap', default='L7')
    a = ap.parse_args()
    (t, x, y, yaw), info = stand_in(a.root, a.lap)
    pd.DataFrame([dict(timestamp_ns=t, x=x, y=y, yaw=yaw)]).to_csv(f'{a.root}/first_pose.csv', index=False)
    json.dump(info, open(f'{a.root}/first_pose_standin.json', 'w'))
    print(f'stand-in first pose ({x:.2f}, {y:.2f}, {np.degrees(yaw):.2f} deg) at {t}: {info["method"]}, '
          f'{info["start_along_ref_m"]:.0f} m along {a.lap}, heading offset {info["heading_offset_deg"]:.2f} deg')
