"""Stand-in first pose for a run without a given first pose (odo.online --no_first_pose; the daytime bag L9).

The whole run's car-state motion (vx, vy, wz integrated over distance) is matched by curvature over distance against
the GT path of a night lap of the map run (default L7), best shift by correlation; the heading offset is the median
over the matched distance; then dead reckoning from the matched start, read at the first image. Inputs: the query
run's allowed car state and image stamps, and the map run's GT (data.gt(): data/extracted or the checkpoints'
train_ref.npz). It uses the whole run (not causal) and only replaces the given first pose of a test run. Its error is a
few metres, so the run starts with a wide initial sigma (init_sigma_xy 3 m, init_sigma_yaw_deg 1 deg).
Port of stand_in_first_pose in visualization/daybag.py (branch praffulr_dev).
"""
import numpy as np

from . import data


def curvature(yaw, ds, window_m=10.0):
    k = np.gradient(yaw, ds)
    w = max(1, int(window_m / ds))
    return np.convolve(k, np.ones(w) / w, mode='same')


def stand_in(root, lap='L7', ds=0.5):
    """((stamp_ns, x, y, yaw), info) for the extracted run at root."""
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
