"""Sweep KF fusion params / camera sets on the 'tune' training lap (never val)."""
import sys, json, itertools; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from odo import fuse_vo, report, data
S = sys.argv[1] if len(sys.argv) > 1 else 'outputs/04_feature_vo_sp/steps_tune'
rows = []
camsets = [['fl'], ['fl', 'fr', 'r'], ['fl', 'fr', 'r', 'cr'], ['fl', 'fr', 'r', 'cr', 'rr'], data.CAMS]
grid = [('kf', g, c, q, x) for g in [True] for c in camsets for q in [0.5, 2.0, 5.0] for x in [9.0, 25.0]] + \
       [('robust', g, c, cs, 0) for g in [False, True] for c in camsets for cs in [1.0, 2.0, 3.0, 5.0]]
for mode, gain, cams, q, chi2 in grid:
    noise = fuse_vo.camera_noise(S, cams, gain_cal=gain)
    if mode == 'kf':
        ts, dyaw, n = fuse_vo.fuse_kf(S, cams, 'tune', noise=noise, q=q, chi2=chi2)
    else:
        ts, dyaw, n = fuse_vo.fuse_robust(S, cams, 'tune', noise=noise, c_scale=q)
    x, y, yaw = fuse_vo.integrate(ts, dyaw)
    m, _ = report.compute_all(pd.DataFrame(dict(timestamp_ns=ts, x=x, y=y, yaw=yaw)), 'tune')
    rows.append(dict(mode=mode, gain=gain, cams='+'.join(cams), q=q, chi2=chi2, pos_rmse=m['pos_rmse'], yaw_rmse=m['yaw_rmse_deg'],
                     rpe1_yaw=m['rpe_1s_yaw_rmse_deg'], rpe10_yaw=m['rpe_10s_yaw_rmse_deg'], rpe10_pct=m['rpe_10s_trans_pct'],
                     no_update=float(np.mean(n == 0) * 100)))
d = pd.DataFrame(rows).sort_values('rpe10_yaw')
d.to_csv(sys.argv[2] if len(sys.argv) > 2 else 'outputs/04_feature_vo_sp/sweep_tune/sweep.csv', index=False)
print(d.head(15).to_string()); print(d.groupby(['mode', 'gain', 'cams'])[['rpe10_yaw', 'pos_rmse']].min())
