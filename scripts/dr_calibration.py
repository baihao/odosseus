"""Dead-reckoning (car-state) drift measured on the training run with its GT: the data that the filter's process noise
(q_lon, q_lat, q0, q_yaw) and the relocalisation plausibility bounds (reloc_sig*, reloc_yaw*) should be derived from,
instead of being tuned.

From GT poses at random start times on the night laps, integrate the allowed car state (odo.online.CarState, same
code as the filter) over a horizon h and compare with GT at t + h, in the car frame at t.
    python scripts/dr_calibration.py -> outputs/24_day/dr_calibration.md"""
import sys

import numpy as np

sys.path.insert(0, '.')
from odo import data
from odo.online import CarState, DEFAULT, move, wrap

rng = np.random.default_rng(0)
cs = CarState(data.allowed_state())
g = data.gt()
tg = g.timestamp_ns.values.astype(np.int64)
t0 = data.t0_ns()
a, b = data.LAPS['L1'][0], data.LAPS['L8'][1]            # racing laps only (out-lap with 80 s in the pit excluded)
starts = rng.uniform(a + 1, b - 11, 3000)
gx, gy, gyaw = data.interp_gt(t0 + (starts * 1e9).astype(np.int64))
L = ['# Car-state dead-reckoning drift vs training GT', '',
     f'{len(starts)} random starts on night laps L1–L8, horizon h; error of the dead-reckoned pose vs GT at t + h, in the '
     'car frame at t. `scripts/dr_calibration.py`.', '',
     '| h [s] | distance median [m] | lon err σ [m] | lat err σ [m] | pos err p95 [m] | yaw err σ [deg] | lon σ per m [%] | lat σ per m [%] | pos p95 per s [m/s] | yaw σ per s [deg/s] |',
     '|---|---|---|---|---|---|---|---|---|---|']
res = {}
for h in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
    lon, lat, dyaw, dist = [], [], [], []
    for k, s in enumerate(starts):
        ta = t0 + int(s * 1e9); tb = ta + int(h * 1e9)
        dx, dy, dth = cs.integrate(ta, tb, tb)
        X = move(np.array([gx[k], gy[k], gyaw[k]]), dx, dy, dth)
        hx, hy, hyaw = data.interp_gt([tb])
        ex, ey = X[0] - hx[0], X[1] - hy[0]
        c, si = np.cos(gyaw[k]), np.sin(gyaw[k])
        lon.append(c * ex + si * ey); lat.append(-si * ex + c * ey); dyaw.append(wrap(X[2] - hyaw[0]))
        dist.append(np.hypot(hx[0] - gx[k], hy[0] - gy[k]))
    lon, lat, dyaw, dist = map(np.array, (lon, lat, dyaw, dist))
    pos = np.hypot(lon, lat)
    m = dist > 1.0
    r = dict(dist=np.median(dist), lon=lon.std(), lat=lat.std(), p95=np.percentile(pos, 95), yaw=np.degrees(dyaw.std()),
             lon_pm=np.std(lon[m] / dist[m]) * 100, lat_pm=np.std(lat[m] / dist[m]) * 100)
    res[h] = r
    L.append(f'| {h:g} | {r["dist"]:.1f} | {r["lon"]:.3f} | {r["lat"]:.3f} | {r["p95"]:.3f} | {r["yaw"]:.3f} | '
             f'{r["lon_pm"]:.2f} | {r["lat_pm"]:.2f} | {r["p95"] / h:.3f} | {r["yaw"] / h:.3f} |')
    print(L[-1], flush=True)

# what the filter's Q predicts for the same horizons (per 0.1 s step: sigma = q * ds + q0 per axis, added in variance;
# yaw: (q_yaw + 0.01 |dtheta|) per step)
q = DEFAULT
L += ['', 'Process noise of the filter (`DEFAULT`) over the same horizons, at the median speed of the samples '
      '(variance added per 0.1 s step, so sigma grows with sqrt(steps)):', '',
      '| h [s] | Q lon σ [m] | Q lat σ [m] | Q yaw σ [deg] (straight) | measured lon / lat σ [m] | measured yaw σ [deg] |',
      '|---|---|---|---|---|---|']
for h, r in res.items():
    n = max(1, int(round(h / 0.1))); ds = r['dist'] / n
    slon = np.sqrt(n) * (q['q_lon'] * ds + q['q0']); slat = np.sqrt(n) * (q['q_lat'] * ds + q['q0'])
    syaw = np.degrees(np.sqrt(n) * q['q_yaw'])
    L.append(f'| {h:g} | {slon:.3f} | {slat:.3f} | {syaw:.3f} | {r["lon"]:.3f} / {r["lat"]:.3f} | {r["yaw"]:.3f} |')
open('outputs/24_day/dr_calibration.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
print('DR_DONE')
