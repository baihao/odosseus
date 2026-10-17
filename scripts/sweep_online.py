"""Parameter sweep of the online localiser on the development laps (tune + dev2; map = train minus the lap +-10 s).
Each config runs both laps; results appended to outputs/21_online/sweep/results.csv.
    python scripts/sweep_online.py NAME '{"n_pass": 5}' [cams...]"""
import sys, json, os, time
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from odo import online, report
name, cfg = sys.argv[1], json.loads(sys.argv[2])
cams = sys.argv[3:] or online.data.CAMS
rows = []
for split in ['tune', 'dev2']:
    out = f'outputs/21_online/sweep/{name}/{split}'
    pred, inf, ms, stat = online.run(split, out, cams=cams, cfg=cfg, verbose=False)
    m, _ = report.compute_all(report.load_pred(f'{out}/pred.csv'), split)
    rows.append(dict(name=name, split=split, cfg=json.dumps(cfg), cams=' '.join(cams), ms=ms, fix=stat.get('fix', 0),
                     **{k: m[k] for k in ['pos_rmse', 'pos_median', 'pos_p95', 'lat_rmse', 'lon_rmse', 'yaw_rmse_deg', 'within_100']}))
    print(rows[-1], flush=True)
r = pd.DataFrame(rows)
f = 'outputs/21_online/sweep/results.csv'
r.to_csv(f, mode='a', header=not os.path.exists(f), index=False)
print('SWEEP_DONE', name, 'mean pos_rmse %.4f lat %.4f yaw %.4f' % (r.pos_rmse.mean(), r.lat_rmse.mean(), r.yaw_rmse_deg.mean()))
