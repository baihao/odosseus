"""Fuse per-camera two-view steps (odo.feature_vo) into a map-frame trajectory.

Per camera: rotation step -> base frame (R_bc R R_bc^T) -> yaw increment; translation direction ->
base frame -> heading-relative motion direction (sideslip-like angle beta).
Cameras run on their own exposure stamps, so per-camera cumulative yaw is resampled onto the output
timeline (fl stamps) and increments are fused with a median over cameras whose step was valid.
Metric step length = integrated car-state speed (allowed input). Causal: output pose k uses camera
frames with stamps <= t_k (+ up to ~1 ms jitter between camera stamps, see note in README).

    python -m odo.fuse_vo --steps outputs/04_feature_vo_sp/steps --cams fl r --out outputs/04_feature_vo_sp/fl_r
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from . import data, report, vo2map


def per_camera(steps_dir, cam, min_inl=30):
    z = np.load(f'{steps_dir}/{cam}.npz')
    Rbc = data.calib()[cam.split('_')[0]]['T_base_cam'][:3, :3]  # 'fl_e' = alternative estimator of cam fl
    Rb = np.einsum('ij,njk,lk->nil', Rbc, z['R'], Rbc)
    dyaw = np.arctan2(Rb[:, 1, 0], Rb[:, 0, 0])
    db = z['t'] @ Rbc.T
    beta = np.arctan2(db[:, 1], db[:, 0])  # direction of camera motion in base frame k-1
    if 'dyaw_base' in z:  # odo.planar_vo output: yaw already in base frame
        dyaw = z['dyaw_base']; beta = np.zeros_like(dyaw)
    ok = z['ok'] & (z['n_inl'] >= min_inl)
    return dict(t_prev=z['t_prev'], t_cur=z['t_cur'], dyaw=dyaw, beta=beta, ok=ok, n_inl=z['n_inl'],
                ms=float(z['ms_per_frame']))


def resample_increments(c, ts):
    """Per-camera yaw increments mapped onto output steps [ts[k-1], ts[k]] by interpolating the
    cumulative yaw; a step is valid only if all camera steps it overlaps were valid."""
    tc = np.concatenate([c['t_prev'][:1], c['t_cur']]).astype(np.float64)
    cum = np.concatenate([[0], np.cumsum(np.where(c['ok'], c['dyaw'], 0.0))])
    bad = np.concatenate([[0], np.cumsum(~c['ok'])])
    y = np.interp(ts.astype(np.float64), tc, cum)
    b = np.interp(ts.astype(np.float64), tc, bad)
    inside = (ts >= tc[0]) & (ts <= tc[-1])
    inc = np.diff(y)
    valid = (np.diff(b) < 1e-9) & inside[1:] & inside[:-1]
    # beta: nearest camera step
    j = np.clip(np.searchsorted(c['t_cur'], ts[1:]), 0, len(c['t_cur']) - 1)
    return inc, valid, c['beta'][j], c['ok'][j]


def camera_noise(steps_dir, cams, gain_cal=True):
    """Per-camera yaw-increment bias and robust sigma (rad/step) vs GT, measured on the steps in
    steps_dir (use the training 'tune' lap, never val)."""
    out = {}
    for cam in cams:
        c = per_camera(steps_dir, cam)
        g0 = data.interp_gt(c['t_prev'])[2]; g1 = data.interp_gt(c['t_cur'])[2]
        gd = np.angle(np.exp(1j * (g1 - g0)))
        ok = c['ok'] & (np.abs(c['dyaw'] - gd) < np.radians(0.5))
        slope, off = np.polyfit(gd[ok], c['dyaw'][ok], 1)  # VO = slope * GT + off (rotation gain error)
        gain = 1.0 / slope if gain_cal else 1.0
        e = (gain * c['dyaw'] - gd)[c['ok']]
        b = float(np.median(e))
        out[cam] = dict(gain=float(gain), bias=b, sigma=float(1.4826 * np.median(np.abs(e - b))),
                        outlier_rate=float(np.mean(np.abs(e - b) > 5 * 1.4826 * np.median(np.abs(e - b)))))
    return out


def fuse_robust(steps_dir, cams, split='val', ref_cam='fl', noise=None, c_scale=3.0, iters=10):
    """Per-step robust fusion (no motion model -> no lag): IRLS with Cauchy weights over the
    gain/bias-corrected camera yaw increments, each scaled by its own sigma. Gap -> repeat last."""
    ts = data.frames_split(ref_cam, split).stamp_ns.values
    Z, V = [], []
    for cam in cams:
        i, v, _, _ = resample_increments(per_camera(steps_dir, cam), ts)
        Z.append(i); V.append(v)
    Z, V = np.array(Z), np.array(V)
    bias = np.array([noise[c]['bias'] if noise else 0.0 for c in cams])[:, None]
    gain = np.array([noise[c].get('gain', 1.0) if noise else 1.0 for c in cams])[:, None]
    sig = np.array([max(noise[c]['sigma'], 1e-4) if noise else np.radians(0.1) for c in cams])[:, None]
    Zc = gain * Z - bias
    W0 = np.where(V, 1.0 / sig ** 2, 0.0)
    Zs = np.where(V, Zc, 0.0)
    # init: weighted median approximated by the median of valid cameras
    est = np.array([np.median(Zc[V[:, k], k]) if V[:, k].any() else np.nan for k in range(Z.shape[1])])
    for _ in range(iters):
        r = (Zs - est[None]) / (c_scale * sig)
        w = W0 / (1 + r ** 2)
        est_new = (w * Zs).sum(0) / np.maximum(w.sum(0), 1e-12)
        est = np.where(V.any(0), est_new, np.nan)
    r = np.abs(Zs - est[None]) / sig
    n_used = ((r < 3) & V).sum(0)
    for k in np.where(np.isnan(est))[0]:
        est[k] = est[k - 1] if k > 0 else 0.0
    return ts, est, n_used


def fuse_kf(steps_dir, cams, split='val', ref_cam='fl', noise=None, q=0.5, chi2=9.0, reset_after=5):
    """Robust KF on yaw rate w [rad/s] (constant-rate model, process noise q [rad/s^2]),
    one scalar update per camera with chi2 gating. Heading integrates the posterior rate."""
    ts = data.frames_split(ref_cam, split).stamp_ns.values
    dt = np.diff(ts) / 1e9
    Z, V = [], []
    for cam in cams:
        i, v, _, _ = resample_increments(per_camera(steps_dir, cam), ts)
        Z.append(i); V.append(v)
    Z, V = np.array(Z), np.array(V)
    bias = np.array([noise[c]['bias'] if noise else 0.0 for c in cams])
    gain = np.array([noise[c].get('gain', 1.0) if noise else 1.0 for c in cams])
    sig = np.array([max(noise[c]['sigma'], 1e-4) if noise else np.radians(0.1) for c in cams])
    w, P = 0.0, 1.0
    dyaw = np.zeros(len(dt)); n_used = np.zeros(len(dt), int); rej_run = 0
    for k in range(len(dt)):
        P += (q * dt[k]) ** 2
        used = 0
        zk = (gain * Z[:, k] - bias) / dt[k]; Rk = (sig / dt[k]) ** 2
        for j in np.argsort(Rk):
            if not V[j, k]:
                continue
            S = P + Rk[j]
            if (zk[j] - w) ** 2 / S < chi2:
                Kg = P / S; w += Kg * (zk[j] - w); P *= (1 - Kg); used += 1
        if used == 0 and V[:, k].any():
            rej_run += 1
            if rej_run >= reset_after:  # filter lost: re-initialise on the camera median
                w = float(np.median(zk[V[:, k]])); P = float(np.median(Rk[V[:, k]])); rej_run = 0
        else:
            rej_run = 0
        n_used[k] = used
        dyaw[k] = w * dt[k]
    return ts, dyaw, n_used


def integrate(ts, dyaw, beta=None):
    dist = vo2map.speed_dist(ts)
    beta = np.zeros(len(dyaw)) if beta is None else beta
    x, y, th = np.zeros(len(ts)), np.zeros(len(ts)), np.zeros(len(ts))
    x[0], y[0], th[0] = data.first_pose(ts[0])
    for k in range(1, len(ts)):
        a = th[k - 1] + 0.5 * dyaw[k - 1] + beta[k - 1]  # chord direction
        x[k] = x[k - 1] + dist[k - 1] * np.cos(a)
        y[k] = y[k - 1] + dist[k - 1] * np.sin(a)
        th[k] = th[k - 1] + dyaw[k - 1]
    return x, y, np.angle(np.exp(1j * th))


def fuse(steps_dir, cams, split='val', ref_cam='fl', beta_mode='vo', beta_cams=None,
         cons_deg=0.3, gate_deg=0.5, max_jump_deg=2.0):
    ts = data.frames_split(ref_cam, split).stamp_ns.values
    incs, vals, betas, bok = [], [], [], []
    for cam in cams:
        c = per_camera(steps_dir, cam)
        i, v, b, bo = resample_increments(c, ts)
        incs.append(i); vals.append(v)
        if beta_cams is None or cam in beta_cams:
            betas.append(b); bok.append(bo)
    I = np.array(incs); V = np.array(vals)
    I_m = np.where(V, I, np.nan)
    dyaw = np.full(I.shape[1], np.nan)
    n_used = np.zeros(I.shape[1], int)
    hist = []
    for k in range(I.shape[1]):
        v = I_m[:, k]; v = v[~np.isnan(v)]
        pred = np.median(hist[-5:]) if hist else 0.0
        est = np.nan
        if len(v):
            med = np.median(v)
            agree = v[np.abs(v - med) < np.radians(cons_deg)]
            if len(agree) >= min(2, len(cams)) or (len(cams) == 1 and abs(med - pred) < np.radians(gate_deg)):
                est = agree.mean()
            # single surviving camera / no consensus: accept only if close to the causal prediction
            elif abs(med - pred) < np.radians(gate_deg):
                est = med
            if not np.isnan(est) and abs(est - pred) > np.radians(max_jump_deg):
                est = np.nan
            n_used[k] = len(agree)
        if np.isnan(est):
            est = pred  # gap: hold recent yaw rate (pure VO, no gyro)
        else:
            hist.append(est)
        dyaw[k] = est
    gap = n_used == 0
    B = np.where(np.array(bok), np.array(betas), np.nan)
    beta = np.nanmedian(B, 0) if beta_mode == 'vo' else np.zeros(len(ts) - 1)
    beta = np.where(np.isnan(beta), 0.0, beta)
    dist = vo2map.speed_dist(ts)
    x, y, th = np.zeros(len(ts)), np.zeros(len(ts)), np.zeros(len(ts))
    x[0], y[0], th[0] = data.first_pose(ts[0])
    for k in range(1, len(ts)):
        # move along the chord direction: heading at step start + beta (+ half the rotation for midpoint)
        a = th[k - 1] + beta[k - 1]
        x[k] = x[k - 1] + dist[k - 1] * np.cos(a)
        y[k] = y[k - 1] + dist[k - 1] * np.sin(a)
        th[k] = th[k - 1] + dyaw[k - 1]
    return ts, x, y, np.angle(np.exp(1j * th)), dict(gap_pct=float(100 * gap.mean()),
                                                     mean_cams_used=float(n_used.mean()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', required=True); ap.add_argument('--cams', nargs='+', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--split', default='val')
    ap.add_argument('--mode', default='robust', choices=['robust', 'kf', 'consensus'])
    ap.add_argument('--c_scale', type=float, default=3.0)
    ap.add_argument('--noise_steps', default=None, help='steps dir of the tune lap for per-camera bias/sigma')
    ap.add_argument('--q', type=float, default=0.5); ap.add_argument('--chi2', type=float, default=9.0)
    ap.add_argument('--beta', default='zero', choices=['vo', 'zero'])
    ap.add_argument('--beta_cams', nargs='*', default=None)
    ap.add_argument('--cons_deg', type=float, default=0.3); ap.add_argument('--gate_deg', type=float, default=0.5)
    ap.add_argument('--max_jump_deg', type=float, default=2.0)
    ap.add_argument('--name', default=None); ap.add_argument('--video', action='store_true')
    a = ap.parse_args()
    if a.mode in ('kf', 'robust'):
        noise = camera_noise(a.noise_steps, a.cams) if a.noise_steps else None
        if a.mode == 'kf':
            ts, dyaw, n_used = fuse_kf(a.steps, a.cams, a.split, noise=noise, q=a.q, chi2=a.chi2)
        else:
            ts, dyaw, n_used = fuse_robust(a.steps, a.cams, a.split, noise=noise, c_scale=a.c_scale)
        x, y, yaw = integrate(ts, dyaw)
        info = dict(mean_cams_used=float(n_used.mean()), no_update_pct=float(100 * np.mean(n_used == 0)), q=a.q,
                    chi2=a.chi2, noise=json.dumps({c: {k: round(v, 5) for k, v in d.items()} for c, d in (noise or {}).items()}))
    else:
        ts, x, y, yaw, info = fuse(a.steps, a.cams, a.split, beta_mode=a.beta, beta_cams=a.beta_cams,
                                   cons_deg=a.cons_deg, gate_deg=a.gate_deg, max_jump_deg=a.max_jump_deg)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(dict(timestamp_ns=ts, x=x, y=y, yaw=yaw)).to_csv(f'{a.out}/pred.csv', index=False, float_format='%.5f')
    ms = sum(per_camera(a.steps, c)['ms'] for c in a.cams)
    extra = dict(method=f'SuperPoint+LightGlue 2-view essential VO, fusion={a.mode}, speed-scaled, beta=0',
                 cams=' '.join(a.cams), runtime_ms_per_frame=ms, **info)
    R = data.ROOT
    m = report.make_report(f'{a.out}/pred.csv', a.name or f'feature VO {"+".join(a.cams)}', a.split, a.video, extra,
                           baselines={'dead-reckoning': f'{R}/outputs/01_deadreckoning/metrics_val.json'},
                           others={'dead-reckoning': f'{R}/outputs/01_deadreckoning/pred_val.csv'})
    print({k: round(v, 3) for k, v in m.items() if isinstance(v, float)})
