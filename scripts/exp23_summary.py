"""Exp23 summary: per configuration, the tuning criterion (L5 under the F1-F4 maps, pooled) and the validation laps
(F1-F3 pooled = 6 laps each once; plus L2 under F2 and L0 under F4 separately). Errors pooled per frame.
Writes outputs/23_odo_weight/summary.csv and prints a markdown table."""
import sys, os, json; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from odo import data, report
root = 'outputs/23_odo_weight'
r = pd.read_csv(f'{root}/results.csv').drop_duplicates(['name', 'fold', 'lap'], keep='last')
def pooled(name, items):
    P = []
    for fold, lap in items:
        p = f'{root}/{name}/{fold}/{lap}/pred.csv'
        if os.path.exists(p):
            P.append(report.compute_all(report.load_pred(p), lap)[1])
    cat = {k: np.concatenate([q[k] for q in P]) for k in ('pos', 'lat', 'dyaw')}
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    return rms(cat['pos']), rms(cat['lat']), rms(cat['dyaw']), float(np.mean(cat['pos'] < 1) * 100), float(cat['pos'].max())
rows = []
for name in r.name.unique():
    cfg = r[r.name == name].cfg.iloc[0]
    tune = pooled(name, [(f, 'L5') for f in ('F1', 'F2', 'F3', 'F4')])
    val6 = pooled(name, [(f, l) for f in ('F1', 'F2', 'F3') for l in data.FOLDS[f]['val']])
    l2 = pooled(name, [('F2', 'L2')]); l0 = pooled(name, [('F4', 'L0')])
    rows.append(dict(name=name, cfg=cfg, L5_pos=tune[0], L5_lat=tune[1], L5_yaw=tune[2], L5_max=tune[4],
                     val6_pos=val6[0], val6_lat=val6[1], val6_yaw=val6[2], val6_w1=val6[3], val6_max=val6[4],
                     L2_pos=l2[0], L2_max=l2[4], L0_pos=l0[0], L0_max=l0[4]))
s = pd.DataFrame(rows); s.to_csv(f'{root}/summary.csv', index=False)
print('| config | change | L5 (tuning) pos / lat / yaw | val 6 laps pos / lat / yaw | val ≤ 1 m | L2 (F2) pos / max | L0 (F4) pos / max |')
print('|---|---|---|---|---|---|---|')
for x in s.itertuples():
    print(f'| {x.name} | `{x.cfg}` | {x.L5_pos:.3f} / {x.L5_lat:.3f} / {x.L5_yaw:.3f}° | {x.val6_pos:.3f} / {x.val6_lat:.3f} / {x.val6_yaw:.3f}° | '
          f'{x.val6_w1:.1f} % | {x.L2_pos:.3f} / {x.L2_max:.2f} | {x.L0_pos:.3f} / {x.L0_max:.2f} |')
