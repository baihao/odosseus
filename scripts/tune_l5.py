"""Tuning on L5 only (the CV tuning lap), under each fold's map. Appends to outputs/21_online/cv_tune/results.csv.
    python scripts/tune_l5.py NAME '{"min_inl": 60}' [F1 F2 F3]"""
import sys, json, os; sys.path.insert(0, '.')
import pandas as pd
from odo import online, report, data
name, cfg = sys.argv[1], json.loads(sys.argv[2]); folds = sys.argv[3:] or list(data.FOLDS)
rows = []
for fold in folds:
    out = f'outputs/21_online/cv_tune/{name}/{fold}'
    pred, inf, ms, stat = online.run('L5', out, cfg=cfg, verbose=False, fold=fold)
    m, _ = report.compute_all(report.load_pred(f'{out}/pred.csv'), 'L5')
    rows.append(dict(name=name, fold=fold, cfg=json.dumps(cfg), ms=ms, **{k: m[k] for k in ['pos_rmse', 'pos_p95', 'lat_rmse', 'lon_rmse', 'yaw_rmse_deg', 'within_100']}))
r = pd.DataFrame(rows); f = 'outputs/21_online/cv_tune/results.csv'
r.to_csv(f, mode='a', header=not os.path.exists(f), index=False)
print('TUNE_DONE', name, ' '.join(f'{a}:{b:.3f}' for a, b in zip(r.fold, r.pos_rmse)), 'mean %.4f lat %.4f yaw %.4f' % (r.pos_rmse.mean(), r.lat_rmse.mean(), r.yaw_rmse_deg.mean()))
