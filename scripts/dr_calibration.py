"""Car-state calibration against the training GT (calibration, not tuning: no localisation error is optimised).

From GT poses at random start times on the night laps, integrate the allowed car state (odo.online.CarState, same
code as the filter) over a horizon h and compare with GT at t + h, in the car frame at t.

Pass 1 (raw car state): systematic errors -> speed scale (median along-track error per metre over 1 s) and yaw-rate
bias (least-squares slope of the mean yaw error over 1-10 s).
Pass 2 (car state corrected by pass 1): the remaining drift sigma vs distance (lon, lat) and time (yaw). Evidence for
the relocalisation plausibility bounds (README: Approach, step 5); the filter's process noise is odo.online.DEFAULT.

    python scripts/dr_calibration.py -> outputs/24_day/dr_calibration.md and outputs/24_day/dr_calibration.json"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
from odo import data
from odo.online import CarState, move, wrap

HS = (0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0)
rng = np.random.default_rng(0)
g = data.gt()
t0 = data.t0_ns()
a, b = data.LAPS['L1'][0], data.LAPS['L8'][1]            # racing laps only (out-lap with 80 s in the pit excluded)
starts = rng.uniform(a + 1, b - 21, 3000)
gx, gy, gyaw = data.interp_gt(t0 + (starts * 1e9).astype(np.int64))


def drift(cs):
    res = {}
    for h in HS:
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
        m = dist > 1.0
        res[h] = dict(dist=np.median(dist), lon=lon.std(), lat=lat.std(), p95=np.percentile(np.hypot(lon, lat), 95),
                      yaw=np.degrees(dyaw.std()), lon_bias_med_pm=np.median(lon[m] / dist[m]),
                      lat_bias_pm=np.mean(lat[m] / dist[m]), yaw_bias=np.mean(dyaw))
        print(f'h {h:g}: lon sigma {res[h]["lon"]:.3f} lat {res[h]["lat"]:.3f} yaw {res[h]["yaw"]:.3f} deg; '
              f'lon bias {res[h]["lon_bias_med_pm"] * 100:+.3f} %, yaw bias {np.degrees(res[h]["yaw_bias"]):+.4f} deg', flush=True)
    return res


# pass 1: systematic errors of the raw car state
r1 = drift(CarState(data.allowed_state()))
speed_scale = float(1.0 / (1.0 + r1[1.0]['lon_bias_med_pm']))
hh = np.array([h for h in HS if 1.0 <= h <= 10.0]); yb = np.array([r1[h]['yaw_bias'] for h in hh])
yaw_rate_bias = float(-(hh @ yb) / (hh @ hh))            # rad/s to ADD to wz
print(f'speed scale {speed_scale:.5f}, yaw-rate correction {np.degrees(yaw_rate_bias):+.4f} deg/s', flush=True)

# pass 2: remaining drift of the corrected car state
cs = CarState(data.allowed_state())
cs.vx, cs.vy, cs.v = cs.vx * speed_scale, cs.vy * speed_scale, cs.v * speed_scale
cs.wz = cs.wz + yaw_rate_bias
r2 = drift(cs)

L = ['# Car-state dead-reckoning drift vs training GT', '',
     f'{len(starts)} random starts on night laps L1–L8, horizon h; error of the dead-reckoned pose vs GT at t + h, in the '
     'car frame at t. `scripts/dr_calibration.py`.', '',
     f'**Systematic errors of the raw car state** (pass 1): distance {(1 / speed_scale - 1) * 100:+.2f} % (median, 1 s), '
     f'yaw-rate bias {-np.degrees(yaw_rate_bias):+.4f} deg/s. Corrections applied by the filter: speed × {speed_scale:.5f}, '
     f'yaw rate {np.degrees(yaw_rate_bias):+.4f} deg/s.', '',
     '| h [s] | distance median [m] | raw: lon σ / lat σ [m] | raw: yaw σ [deg] | raw: lon bias [% of dist.] | raw: yaw bias [deg] | **corrected: lon σ / lat σ [m]** | **corrected: yaw σ [deg]** | corrected: pos p95 [m] | corrected: lon bias [%] |',
     '|---|---|---|---|---|---|---|---|---|---|']
for h in HS:
    p, q = r1[h], r2[h]
    L.append(f'| {h:g} | {q["dist"]:.1f} | {p["lon"]:.3f} / {p["lat"]:.3f} | {p["yaw"]:.3f} | {p["lon_bias_med_pm"] * 100:+.3f} | '
             f'{np.degrees(p["yaw_bias"]):+.4f} | {q["lon"]:.3f} / {q["lat"]:.3f} | {q["yaw"]:.3f} | {q["p95"]:.3f} | '
             f'{q["lon_bias_med_pm"] * 100:+.3f} |')
open('outputs/24_day/dr_calibration.md', 'w').write('\n'.join(L) + '\n')
json.dump({'source': 'scripts/dr_calibration.py: car-state dead reckoning vs training GT, night laps L1-L8, '
                     f'{len(starts)} random starts; sigma table measured after the speed / yaw-rate correction',
           'speed_scale': speed_scale, 'yaw_rate_bias_radps': yaw_rate_bias, 'h_s': list(HS),
           'dist_m': [float(r2[h]['dist']) for h in HS], 'sigma_lon_m': [float(r2[h]['lon']) for h in HS],
           'sigma_lat_m': [float(r2[h]['lat']) for h in HS],
           'sigma_yaw_rad': [float(np.radians(r2[h]['yaw'])) for h in HS]},
          open('outputs/24_day/dr_calibration.json', 'w'), indent=1)
print('\n'.join(L))
print('DR_DONE')
