"""Build a fake 'test run' folder from a time window of the training run (default: val laps, t >= 897 s) in the layout of scripts/extract_mcap.py, without
GT except first_pose.csv. Checks the test-mode plumbing (odo.online --split test --query ...)."""
import sys, os, json, shutil; sys.path.insert(0, '.')
import pandas as pd
from odo import data
out = sys.argv[1] if len(sys.argv) > 1 else 'data/fake_test'
ta_s, tb_s = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (897.0, 1e9)
os.makedirs(f'{out}/images', exist_ok=True)
t0 = data.t0_ns(); ta = t0 + int(ta_s * 1e9); tb = t0 + int(tb_s * 1e9)
for c in data.CAMS:
    f = pd.read_csv(f'{data.DATA}/images/{c}.csv')
    f = f[(f.stamp_ns >= ta) & (f.stamp_ns < tb)].copy()
    f['file'] = os.path.relpath(data.DATA, out) + '/' + f.file          # point at the extracted images, no copy
    f.to_csv(f'{out}/images/{c}.csv', index=False)
s = pd.read_csv(f'{data.DATA}/state.csv')
st = s['header.stamp.sec'].astype('int64') * 10**9 + s['header.stamp.nanosec'].astype('int64')
keep = ['header.stamp.sec', 'header.stamp.nanosec'] + [c for c in data.ALLOWED_STATE if c != 'timestamp_ns']
s[(st >= ta - int(1e8)) & (st < tb)][keep].to_csv(f'{out}/state.csv', index=False)
shutil.copy(f'{data.DATA}/calib.json', f'{out}/calib.json')
g = data.gt(); r = g[g.timestamp_ns >= ta].iloc[0]
pd.DataFrame([dict(timestamp_ns=int(r.timestamp_ns), x=r.x, y=r.y, yaw=r.yaw)]).to_csv(f'{out}/first_pose.csv', index=False)
print('fake test run in', out)
