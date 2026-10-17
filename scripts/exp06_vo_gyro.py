"""Exp06: gyro (car-state wz) with bias tracked from the exp05 VO; translation from car-state vx, vy."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, os
from odo import fuse_vo, vo_gyro, report, data
cams = [c + s for c in data.CAMS for s in ['_p', '_e']]
noise = fuse_vo.camera_noise('outputs/05_planar_vo/merged_tune', cams)
ts, dyaw, n = fuse_vo.fuse_robust('outputs/05_planar_vo/merged_val', cams, 'val', noise=noise, c_scale=1.0)
out = 'outputs/06_vo_gyro'; os.makedirs(out, exist_ok=True)
tau, rmax = 80.0, 0.2  # best non-trivial setting on the tune lap
dy, b = vo_gyro.fuse(ts, dyaw, n >= 2, tau, rmax)
x, y, yaw = vo_gyro.integrate(ts, dy)
pd.DataFrame(dict(timestamp_ns=ts, x=x, y=y, yaw=yaw)).to_csv(f'{out}/pred.csv', index=False, float_format='%.5f')
np.save(f'{out}/gyro_bias.npy', b)
R = data.ROOT
m = report.make_report(f'{out}/pred.csv', 'Exp06 gyro + VO bias tracking (tau 80 s)', 'val', True,
                       dict(method='car-state wz minus bias tracked from planar+essential VO; vx,vy translation',
                            tau_s=tau, r_max_deg_s=rmax),
                       baselines={'dead-reckoning': f'{R}/outputs/01_deadreckoning/metrics_val.json',
                                  'VO exp05': f'{R}/outputs/05_planar_vo/fuse_val/metrics_val.json'},
                       others={'dead-reckoning': f'{R}/outputs/01_deadreckoning/pred_val.csv'})
print({k: round(v, 3) for k, v in m.items() if isinstance(v, float)})
