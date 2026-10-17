"""Retrieval hyper-parameter / camera-set sweep on the tune lap (db = train minus tune lap)."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from odo import retrieval, data, report
model = sys.argv[1] if len(sys.argv) > 1 else 'megaloc'
D = 'outputs/10_vpr_desc'
rows = []
camsets = [['fl'], ['fr'], ['r'], ['cl'], ['cr'], ['rl'], ['rr'], ['fl', 'fr'], ['fl', 'fr', 'r'], ['fr', 'r', 'cl', 'cr'],
           ['fl', 'fr', 'r', 'rl', 'rr', 'cl', 'cr']]
for cams in camsets:
    tq, Dq, tdb, Ddb = retrieval.query_db(D, model, cams, 'tune')
    S = retrieval.similarity(Dq, Ddb)
    db = data.interp_gt(tdb); gx, gy, gyaw = data.interp_gt(tq)
    for mode, k in [('top1', 1), ('topk', 3), ('topk', 5), ('topk', 10)]:
        x, y, yaw, _ = retrieval.estimate(S, db, k, 10.0, 0.02, mode)
        e = np.hypot(x - gx, y - gy)
        m, _ = report.compute_all(pd.DataFrame(dict(timestamp_ns=tq, x=x, y=y, yaw=yaw)), 'tune')
        rows.append(dict(model=model, cams='+'.join(cams), mode=mode, k=k, median=np.median(e), pos_rmse=m['pos_rmse'],
                         lat_rmse=m['lat_rmse'], lon_rmse=m['lon_rmse'], yaw_rmse=m['yaw_rmse_deg'],
                         **retrieval.recall_at(e)))
d = pd.DataFrame(rows); d.to_csv(f'outputs/10_vpr_desc/sweep_tune_{model}.csv', index=False)
pd.set_option('display.width', 250)
print(d.sort_values('pos_rmse').head(12).round(2).to_string())
print(d[d['mode'] == 'top1'].round(2).to_string())
