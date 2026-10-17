"""Metrics. `challenge_metrics` reproduces data/.../tools/evaluate.py exactly; the rest are
VO-specific diagnostics (drift-independent relative errors, aligned ATE)."""
import numpy as np

from . import data

MAX_GT_GAP_NS = 50_000_000
BIN_NS = 1_000_000_000


def wrap(a):
    return np.angle(np.exp(1j * np.asarray(a)))


def challenge_metrics(ts, x, y, yaw, t_range=None):
    """Same computation as tools/evaluate.py. Coverage is computed over t_range (ns) if given,
    otherwise over the whole GT run."""
    g = data.gt()
    tg = g.timestamp_ns.values
    ts = np.asarray(ts, dtype=np.int64)
    o = np.argsort(ts); ts, x, y, yaw = ts[o], np.asarray(x)[o], np.asarray(y)[o], np.asarray(yaw)[o]
    i = np.searchsorted(tg, ts)
    ok = (i > 0) & (i < len(tg))
    i = np.clip(i, 1, len(tg) - 1)
    ok &= (tg[i] - tg[i - 1]) <= MAX_GT_GAP_NS
    n_sub = len(ts)
    ts, i, sx, sy, syaw = ts[ok], i[ok], x[ok], y[ok], yaw[ok]
    a = (ts - tg[i - 1]) / (tg[i] - tg[i - 1])
    yaw_u = np.unwrap(g.yaw.values)
    gx = g.x.values[i - 1] + a * (g.x.values[i] - g.x.values[i - 1])
    gy = g.y.values[i - 1] + a * (g.y.values[i] - g.y.values[i - 1])
    gyaw = yaw_u[i - 1] + a * (yaw_u[i] - yaw_u[i - 1])
    dx, dy = sx - gx, sy - gy
    lon = np.cos(gyaw) * dx + np.sin(gyaw) * dy
    lat = -np.sin(gyaw) * dx + np.cos(gyaw) * dy
    pos = np.hypot(dx, dy)
    dyaw = np.degrees(wrap(syaw - gyaw))
    lo, hi = (tg[0], tg[-1]) if t_range is None else t_range
    bins = np.arange(lo, hi, BIN_NS)
    covered = np.unique(np.searchsorted(bins, ts, side='right') - 1)
    covered = covered[(covered >= 0) & (covered < len(bins))]
    rms = lambda v: float(np.sqrt(np.mean(v ** 2))) if len(v) else float('nan')
    m = dict(n_submitted=n_sub, n_evaluated=int(len(ts)), coverage_pct=100 * len(covered) / max(len(bins), 1),
             pos_rmse=rms(pos), pos_median=float(np.median(pos)), pos_p95=float(np.percentile(pos, 95)),
             lat_rmse=rms(lat), lon_rmse=rms(lon), yaw_rmse_deg=rms(dyaw),
             within_025=float(np.mean(pos < .25) * 100), within_050=float(np.mean(pos < .5) * 100),
             within_100=float(np.mean(pos < 1.) * 100))
    per = dict(ts=ts, gx=gx, gy=gy, gyaw=wrap(gyaw), x=sx, y=sy, yaw=syaw, pos=pos, lat=lat, lon=lon, dyaw=dyaw)
    return m, per


def relative_errors(ts, x, y, yaw, deltas_s=(1.0, 10.0)):
    """Drift-independent VO quality: for every pose pair (i, i+Δ) compare the relative motion
    expressed in the frame of pose i with the GT relative motion.
    Returns per Δ: translation RMSE [m], translation error as % of distance, yaw RMSE [deg]."""
    ts = np.asarray(ts, dtype=np.int64)
    gx, gy, gyaw = data.interp_gt(ts)
    out = {}
    dt = np.median(np.diff(ts)) / 1e9
    for d in deltas_s:
        k = max(1, int(round(d / dt)))
        if len(ts) <= k:
            continue
        def rel(X, Y, Th):
            ddx, ddy = X[k:] - X[:-k], Y[k:] - Y[:-k]
            c, s = np.cos(Th[:-k]), np.sin(Th[:-k])
            return c * ddx + s * ddy, -s * ddx + c * ddy, wrap(Th[k:] - Th[:-k])
        px, py, pth = rel(np.asarray(x), np.asarray(y), np.asarray(yaw))
        qx, qy, qth = rel(gx, gy, gyaw)
        et = np.hypot(px - qx, py - qy)
        dist = np.hypot(qx, qy)
        mv = dist > 0.5 * d  # ignore standstill
        out[f'rpe_{d:g}s_trans_rmse_m'] = float(np.sqrt(np.mean(et[mv] ** 2)))
        out[f'rpe_{d:g}s_trans_pct'] = float(100 * np.sum(et[mv]) / max(np.sum(dist[mv]), 1e-9))
        out[f'rpe_{d:g}s_yaw_rmse_deg'] = float(np.degrees(np.sqrt(np.mean(wrap(pth - qth)[mv] ** 2))))
    return out


def align_se2(ts, x, y, yaw, scale=False):
    """Umeyama SE(2)/Sim(2) alignment of a trajectory to GT (diagnostic only, not causal)."""
    gx, gy, _ = data.interp_gt(ts)
    P = np.stack([x, y]); Q = np.stack([gx, gy])
    mp, mq = P.mean(1, keepdims=True), Q.mean(1, keepdims=True)
    Pc, Qc = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(Qc @ Pc.T)
    D = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (Pc ** 2).sum() if scale else 1.0
    A = s * R @ Pc + mq
    th = np.arctan2(R[1, 0], R[0, 0])
    return A[0], A[1], wrap(np.asarray(yaw) + th), s


def ate(ts, x, y, yaw, scale=False):
    ax, ay, _, s = align_se2(ts, x, y, yaw, scale)
    gx, gy, _ = data.interp_gt(ts)
    e = np.hypot(ax - gx, ay - gy)
    return float(np.sqrt(np.mean(e ** 2))), s
