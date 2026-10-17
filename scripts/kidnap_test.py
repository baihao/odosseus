"""Robustness: inject a pose error into the filter (kidnap) and measure the time until the error is < 0.3 m again.
    python scripts/kidnap_test.py dev2 30 5 2 3   (t since start [s], lon [m], lat [m], yaw [deg])"""
import sys, json; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from odo import online, report, data
split = sys.argv[1]; t_k, lon, lat, yaw = map(float, sys.argv[2:6])
extra = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
name = f'kidnap_{split}_{t_k:g}_{lon:g}_{lat:g}_{yaw:g}' + ('_' + '_'.join(f'{k}{v}' for k, v in extra.items()) if extra else '')
out = f'outputs/21_online/robustness/{name}'
cfg = dict(extra_dof=3, kidnap=[t_k, lon, lat, yaw], **extra)
pred, inf, ms, stat = online.run(split, out, cfg=cfg, verbose=False, limit=int((t_k + 40) * 10))
p = report.load_pred(f'{out}/pred.csv'); gx, gy, _ = data.interp_gt(p.timestamp_ns.values)
e = np.hypot(p.x - gx, p.y - gy).values; t = (p.timestamp_ns.values - p.timestamp_ns.values[0]) / 1e9
after = t >= t_k
rec = np.where(after & (e < 0.3) & (t > t_k + 0.05))[0]
first_ok = None
for i in rec:            # recovered = stays < 0.3 m for 1 s
    w = (t >= t[i]) & (t < t[i] + 1.0)
    if (e[w] < 0.3).all():
        first_ok = t[i] - t_k; break
print(f'KIDNAP {name}: error right after {e[after][0]:.2f} m, recovered (<0.3 m for 1 s) after '
      f'{first_ok if first_ok is not None else float("nan"):.1f} s; max err after recovery '
      f'{e[t > t_k + (first_ok or 1e9)].max() if first_ok is not None else float("nan"):.2f} m; statuses {inf.status[after].value_counts().to_dict()}')
