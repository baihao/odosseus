"""List the N highest-error moments of an online run (distinct peaks >= 5 s apart) with context:
error split, speed, yaw rate, filter status, matches, GT accuracy flag, and whether the same place is bad on the other lap.
    python scripts/val_high_error.py outputs/21_online/val val 5"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from odo import data, report
d, split, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
p = report.load_pred(f'{d}/pred.csv'); m, per = report.compute_all(p, split)
inf = pd.read_csv(f'{d}/online_info.csv').set_index('timestamp_ns').loc[per['ts']]
st = data.interp_state(per['ts'], ('v_mps', 'wz_radps', 'ax_mps2', 'ay_mps2'))
sd = data.state(); acc = np.hypot(sd['pos_accuracy.0'], sd['pos_accuracy.1']).values
t = (per['ts'] - data.t0_ns()) / 1e9
order = np.argsort(-per['pos']); peaks = []
for i in order:
    if all(abs(t[i] - t[j]) > 5 for j in peaks):
        peaks.append(i)
    if len(peaks) == n:
        break
rows = []
for i in sorted(peaks):
    w = (t > t[i] - 1) & (t < t[i] + 1)
    j = np.searchsorted(sd.timestamp_ns.values, per['ts'][i])
    gacc = acc[max(j - 300, 0):j + 300].max()                      # GT accuracy flag within +-3 s
    # same place on the other val lap: nearest GT point of a pass >= 50 s away
    dxy = np.hypot(per['gx'] - per['gx'][i], per['gy'] - per['gy'][i]); dxy[np.abs(t - t[i]) < 50] = np.inf
    k = int(np.argmin(dxy))
    rows.append(dict(t_s=round(t[i], 1), timestamp_ns=int(per['ts'][i]), x=round(per['gx'][i], 1), y=round(per['gy'][i], 1),
                     pos_err_m=round(per['pos'][i], 3), lat_m=round(per['lat'][i], 3), lon_m=round(per['lon'][i], 3),
                     yaw_err_deg=round(per['dyaw'][i], 3), err_gt_0p25m_s=round(w.sum() * 0.1 * np.mean(per['pos'][w] > 0.25), 1),
                     v_mps=round(st['v_mps'][i], 1), wz_degps=round(np.degrees(st['wz_radps'][i]), 1), ay_mps2=round(st['ay_mps2'][i], 1),
                     ax_mps2=round(st['ax_mps2'][i], 1), status=inf.status.iloc[i], inliers=int(inf.n_inl.iloc[i]),
                     inl_per_cam=inf.inl_cam.iloc[i], dpitch_deg=round(inf.dpitch.iloc[i], 2), droll_deg=round(inf.droll.iloc[i], 2),
                     gt_acc_max_3s=round(gacc, 2), other_lap_t_s=round(t[k], 1), other_lap_dist_m=round(dxy[k], 1),
                     other_lap_err_m=round(per['pos'][k], 3)))
r = pd.DataFrame(rows); pd.set_option('display.width', 300); print(r.to_string(index=False))
r.to_csv(f'{d}/high_error_moments.csv', index=False)
