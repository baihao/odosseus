"""Exp 01: dead reckoning from allowed car state (vx, vy, wz), anchored at the segment's first pose.
Reference for what VO must beat; also validates the metric code against tools/evaluate.py."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from odo import data, report
os.makedirs('outputs/01_deadreckoning', exist_ok=True)

def dead_reckon(ts_out, split, use_vy=True, yaw_src='wz'):
    a, b = data.SPLITS[split]; t0 = data.t0_ns()
    s = data.allowed_state()
    s = s[(s.timestamp_ns >= t0 + a * 1e9 - 2e8) & (s.timestamp_ns < t0 + b * 1e9 + 2e8)]
    tt = s.timestamp_ns.values; dt = np.diff(tt) / 1e9
    wz = s.wz_radps.values; vx = s.vx_mps.values; vy = s.vy_mps.values if use_vy else 0 * vx
    x0, y0, th0 = data.first_pose(ts_out[0])
    # integrate from the first output stamp
    k0 = np.searchsorted(tt, ts_out[0])
    th = np.zeros(len(tt)); X = np.zeros(len(tt)); Y = np.zeros(len(tt))
    th[k0], X[k0], Y[k0] = th0, x0, y0
    for k in range(k0, len(tt) - 1):
        thm = th[k] + 0.5 * wz[k] * dt[k]
        th[k + 1] = th[k] + 0.5 * (wz[k] + wz[k + 1]) * dt[k]
        vxm, vym = 0.5 * (vx[k] + vx[k + 1]), 0.5 * (vy[k] + vy[k + 1])
        X[k + 1] = X[k] + (np.cos(thm) * vxm - np.sin(thm) * vym) * dt[k]
        Y[k + 1] = Y[k] + (np.sin(thm) * vxm + np.cos(thm) * vym) * dt[k]
    # causal: pose at image stamp = last state sample <= stamp + linear extrapolation is avoided; use interp of
    # samples up to the stamp (state at 106 Hz, the sample right after the stamp is < 10 ms later)
    out = lambda v: np.interp(ts_out, tt[k0:], v[k0:])
    return out(X), out(Y), np.angle(np.exp(1j * out(np.unwrap(th))))

for split in ['val', 'all']:
    f = data.frames_split('fl', split)
    ts = f.stamp_ns.values
    x, y, yaw = dead_reckon(ts, split)
    p = f'outputs/01_deadreckoning/pred_{split}.csv'
    pd.DataFrame(dict(timestamp_ns=ts, x=x, y=y, yaw=yaw)).to_csv(p, index=False, float_format='%.5f')
    import shutil; shutil.copy(p, 'outputs/01_deadreckoning/pred.csv')
    m = report.make_report('outputs/01_deadreckoning/pred.csv', f'Exp01 dead reckoning (vx,vy,wz) [{split}]',
                           split, video=(split == 'val'),
                           extra=dict(method='integrate car-state vx,vy,wz from GT pose at segment start'))
    print(split, {k: round(v, 3) for k, v in m.items() if isinstance(v, float)})
# sanity: GT as prediction must give ~0
f = data.frames_split('fl', 'val'); gx, gy, gyaw = data.interp_gt(f.stamp_ns.values)
pd.DataFrame(dict(timestamp_ns=f.stamp_ns.values, x=gx, y=gy, yaw=gyaw)).to_csv('outputs/01_deadreckoning/gt_as_pred.csv', index=False, float_format='%.5f')
