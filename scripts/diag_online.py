"""Per-2-second diagnostics of an online run: error, status mix, matches/inliers, measurement error, speed."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from odo import data
d = sys.argv[1]
p = pd.read_csv(f'{d}/pred.csv'); inf = pd.read_csv(f'{d}/online_info.csv')
gx, gy, gyaw = data.interp_gt(p.timestamp_ns.values)
e = np.hypot(p.x - gx, p.y - gy)
lat = -np.sin(gyaw) * (p.x - gx) + np.cos(gyaw) * (p.y - gy)
me = np.hypot(inf.get('mx', np.nan) - gx, inf.get('my', np.nan) - gy)
v = data.interp_state(p.timestamp_ns.values, ('v_mps',))['v_mps']
t = (p.timestamp_ns - data.t0_ns()) / 1e9
df = pd.DataFrame(dict(t=t, e=e, lat=lat, me=me, st=inf.status, m1=inf.n_m1, inl=inf.n_inl, d2=inf.get('d2'), v=v, ms=inf.ms))
df['bin'] = (df.t // float(sys.argv[2] if len(sys.argv) > 2 else 2)).astype(int)
g = df.groupby('bin').agg(t=('t', 'first'), e=('e', 'max'), me_med=('me', 'median'), fix=('st', lambda s: (s == 'fix').mean()),
                          gated=('st', lambda s: (s == 'gated').mean()), m1=('m1', 'median'), inl=('inl', 'median'),
                          d2=('d2', 'median'), v=('v', 'mean'), ms=('ms', 'mean'))
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 500)
print(g.round(2).to_string())
print('raw measurement error (when available): median %.3f, rmse %.3f' % (np.nanmedian(me), np.sqrt(np.nanmean(me ** 2))))
