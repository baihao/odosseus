"""VO + car-state fusion (the 'motion model' use of VO from the brief).

Yaw: gyro increments (car-state wz, allowed) minus a bias b that is tracked causally from the
difference gyro - VO with a robust first-order filter (time constant tau):
    r_k = (gyro_k - vo_k) - b_{k-1};   b_k = b_{k-1} + (dt/tau) * clip(r_k, +-r_max)   (only if VO valid)
Translation: car-state vx, vy (as in dead reckoning) rotated by the corrected heading.
"""
import numpy as np

from . import data


def gyro_increments(ts):
    s = data.allowed_state()
    tt = s.timestamp_ns.values.astype(np.float64)
    cum = np.concatenate([[0], np.cumsum(0.5 * (s.wz_radps.values[1:] + s.wz_radps.values[:-1]) * np.diff(tt) / 1e9)])
    return np.diff(np.interp(np.asarray(ts, dtype=np.float64), tt, cum))


def body_increments(ts):
    """Integrated vx, vy over each step in the body frame (small-angle, same as dead reckoning)."""
    s = data.allowed_state()
    tt = s.timestamp_ns.values.astype(np.float64)
    out = []
    for c in ('vx_mps', 'vy_mps'):
        v = s[c].values
        cum = np.concatenate([[0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(tt) / 1e9)])
        out.append(np.diff(np.interp(np.asarray(ts, dtype=np.float64), tt, cum)))
    return out


def fuse(ts, vo_dyaw, vo_ok, tau=30.0, r_max_deg_s=0.5):
    g = gyro_increments(ts)
    dt = np.diff(ts) / 1e9
    b = 0.0  # rad/s
    dyaw = np.zeros(len(g)); bias = np.zeros(len(g))
    rmax = np.radians(r_max_deg_s)
    for k in range(len(g)):
        if vo_ok[k] and dt[k] > 0:
            r = (g[k] - vo_dyaw[k]) / dt[k] - b
            b += min(dt[k] / tau, 1.0) * np.clip(r, -rmax, rmax)
        bias[k] = b
        dyaw[k] = g[k] - b * dt[k]
    return dyaw, bias


def integrate(ts, dyaw):
    dx, dy = body_increments(ts)
    x, y, th = np.zeros(len(ts)), np.zeros(len(ts)), np.zeros(len(ts))
    x[0], y[0], th[0] = data.first_pose(ts[0])
    for k in range(1, len(ts)):
        a = th[k - 1] + 0.5 * dyaw[k - 1]
        x[k] = x[k - 1] + np.cos(a) * dx[k - 1] - np.sin(a) * dy[k - 1]
        y[k] = y[k - 1] + np.sin(a) * dx[k - 1] + np.cos(a) * dy[k - 1]
        th[k] = th[k - 1] + dyaw[k - 1]
    return x, y, np.angle(np.exp(1j * th))
