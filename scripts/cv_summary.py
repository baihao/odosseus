"""Aggregate the Exp21 cross-validation: per lap, per fold (pooled over its validation laps), the 6 validation laps of
F1-F3 pooled (each lap once), and the tuning lap L5 under each fold's map. Also writes the per-lap GT files for tools/evaluate.py.
    python scripts/cv_summary.py"""
import sys, json, os; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from odo import data, report
root = 'outputs/21_online/cv'
rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
def summ(per, n_laps):
    return dict(n=len(per['pos']), pos_rmse=rms(per['pos']), pos_median=float(np.median(per['pos'])), pos_p95=float(np.percentile(per['pos'], 95)),
                lat_rmse=rms(per['lat']), lon_rmse=rms(per['lon']), yaw_rmse_deg=rms(per['dyaw']),
                within_025=100 * float(np.mean(per['pos'] < .25)), within_050=100 * float(np.mean(per['pos'] < .5)),
                within_100=100 * float(np.mean(per['pos'] < 1)), max_err=float(per['pos'].max()))
rows, pooled = [], {}
g = data.gt(); t0 = data.t0_ns()
for fold, f in data.FOLDS.items():
    for lap in f['val'] + [data.tune_lap(fold)]:
        d = f'{root}/{fold}/{lap}'
        if not os.path.exists(f'{d}/pred.csv'):
            print('missing', d); continue
        m, per = report.compute_all(report.load_pred(f'{d}/pred.csv'), lap)
        inf = pd.read_csv(f'{d}/online_info.csv')
        role = 'tuning' if lap == data.tune_lap(fold) else 'validation'
        rows.append(dict(fold=fold, map=f['map'], lap=lap, role=role, **summ(per, 1), coverage=m['coverage_pct'],
                         fix_pct=100 * float((inf.status == 'fix').mean()), ms=float(inf.ms.iloc[1:].mean())))
        pooled.setdefault((fold, role), []).append(per)
        a, b = data.LAPS[lap]
        gl = g[(g.timestamp_ns >= t0 + a * 1e9) & (g.timestamp_ns < t0 + b * 1e9)][['timestamp_ns', 'x', 'y', 'yaw']]
        gl.to_csv(f'{d}/gt_{lap}.csv', index=False)
cat = lambda pers: {k: np.concatenate([p[k] for p in pers]) for k in ('pos', 'lat', 'lon', 'dyaw')}
for (fold, role), pers in pooled.items():
    if role == 'validation':
        rows.append(dict(fold=fold, map=data.FOLDS[fold]['map'], lap='+'.join(data.FOLDS[fold]['val']), role='validation (fold pooled)', **summ(cat(pers), 2)))
allv = [p for (fo, r), ps in pooled.items() if r == 'validation' and fo in ('F1', 'F2', 'F3') for p in ps]
rows.append(dict(fold='F1+F2+F3', map='-', lap='L1 L2 L3 L4 L6 L7', role='validation (6 laps pooled, each once)', **summ(cat(allv), 6)))
tun = [p for (fo, r), ps in pooled.items() if r == 'tuning' and fo in ('F1', 'F2', 'F3', 'F4') for p in ps]
rows.append(dict(fold='all folds', map='-', lap='L5 x %d maps' % len(tun), role='tuning (pooled)', **summ(cat(tun), len(tun))))
r = pd.DataFrame(rows)
r.to_csv(f'{root}/cv_summary.csv', index=False)
pd.set_option('display.width', 250)
print(r.drop(columns=['map']).round(3).to_string(index=False))
