"""Is the EKF consistent? Statistics that decide whether thresholds can be derived from the filter's own covariances.

* NIS = d2 of each fix (innovation^T S^-1 innovation, 3 DoF): mean 3 and ~0.1 % above the gate 16 if the filter and the
  measurement covariances are right.
* NEES (position only) = (err_x / sx)^2 + (err_y / sy)^2 against GT: mean ~2 if P is right (x/y correlation ignored).
* dead-reckoning drift per metre and per second between consecutive accepted fixes vs what Q predicts.
    python scripts/filter_consistency.py <run_dir> ...  -> logs + outputs/24_day/filter_consistency.md"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from odo import data
from odo.online import DEFAULT

rows = []
for run in sys.argv[1:]:
    inf = pd.read_csv(f'{run}/online_info.csv')
    p = pd.read_csv(f'{run}/pred.csv')
    gx, gy, _ = data.interp_gt(p.timestamp_ns.values)
    ex, ey = p.x.values - gx, p.y.values - gy
    nees = (ex / inf.sx.values) ** 2 + (ey / inf.sy.values) ** 2
    st = inf.status.fillna('none')
    fix = st.isin(['fix'])
    d2 = inf.d2[fix | st.isin(['gated'])].dropna()
    inn = inf.innov[fix].dropna()
    sig = np.hypot(inf.sx, inf.sy)
    rows.append(dict(run=run, frames=len(p), pos_rmse=float(np.sqrt(np.mean(ex ** 2 + ey ** 2))),
                     nis_mean=float(d2.mean()), nis_median=float(d2.median()), nis_gt16_pct=float(np.mean(d2 > 16) * 100),
                     nees_mean=float(np.mean(nees)), nees_median=float(np.median(nees)),
                     nees_gt_chi2_99_pct=float(np.mean(nees > 9.21) * 100),
                     sigma_xy_median=float(np.median(sig)), err_median=float(np.median(np.hypot(ex, ey))),
                     innov_median=float(inn.median()), rms_px_median=float(inf.rms_px[fix].median())))
    print(json.dumps(rows[-1]), flush=True)

L = ['# EKF consistency on existing runs', '',
     'NIS = d2 of fixes reaching the gate (expected mean 3 for 3 DoF, ~0.1 % above 16 if consistent). NEES = position '
     'error normalised by the filter sigma (x, y; expected mean ~2, ~1 % above 9.21). `scripts/filter_consistency.py`.', '',
     '| run | pos RMSE [m] | NIS mean / median | NIS > 16 [%] | NEES mean / median | NEES > 9.21 [%] | filter σ median [m] | error median [m] | innovation median [m] | reproj. rms median [px] |',
     '|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
    L.append(f'| `{r["run"]}` | {r["pos_rmse"]:.3f} | {r["nis_mean"]:.1f} / {r["nis_median"]:.1f} | {r["nis_gt16_pct"]:.1f} | '
             f'{r["nees_mean"]:.1f} / {r["nees_median"]:.1f} | {r["nees_gt_chi2_99_pct"]:.1f} | {r["sigma_xy_median"]:.3f} | '
             f'{r["err_median"]:.3f} | {r["innov_median"]:.3f} | {r["rms_px_median"]:.2f} |')
open('outputs/24_day/filter_consistency.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
print('CONSIST_DONE')
