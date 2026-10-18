"""Exp23: does relying more on car-state odometry reduce the error? One filter configuration, run on every validation
lap of F1-F4 and on the tuning lap L5 under each of their maps. Appends to outputs/23_odo_weight/results.csv.
Selection criterion: L5 (tuning). Validation laps are reported for analysis only.
    python scripts/exp23_odo_weight.py NAME '{"r0_pos": 0.2, "r0_yaw": 0.008}'"""
import sys, json, os; sys.path.insert(0, '.')
import pandas as pd
from odo import online, report, data
name, cfg = sys.argv[1], json.loads(sys.argv[2])
rows = []
for fold in ['F1', 'F2', 'F3', 'F4']:
    for lap in data.FOLDS[fold]['val'] + [data.tune_lap(fold)]:
        out = f'outputs/23_odo_weight/{name}/{fold}/{lap}'
        pred, inf, ms, stat = online.run(lap, out, cfg=cfg, verbose=False, fold=fold)
        m, _ = report.compute_all(report.load_pred(f'{out}/pred.csv'), lap)
        rows.append(dict(name=name, cfg=json.dumps(cfg), fold=fold, lap=lap, role='tuning' if lap == data.tune_lap(fold) else 'validation',
                         fix_pct=stat.get('fix', 0), **{k: m[k] for k in ['pos_rmse', 'pos_p95', 'lat_rmse', 'lon_rmse', 'yaw_rmse_deg',
                                                                         'within_100', 'rpe_1s_trans_rmse_m', 'rpe_1s_yaw_rmse_deg']}))
        print(rows[-1], flush=True)
r = pd.DataFrame(rows); f = 'outputs/23_odo_weight/results.csv'
r.to_csv(f, mode='a', header=not os.path.exists(f), index=False)
print('EXP23_DONE', name)
