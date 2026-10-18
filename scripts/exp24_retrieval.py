"""Exp24: global retrieval (MegaLoc, odo.online.Reloc) on night vs day-proxy query images: distance of the top-1 and of the
best of the 4 verified candidates to the GT, every 5th image set of a lap, database = the fold's night map laps.
    python scripts/exp24_retrieval.py --fold F5 --lap L2 [--every 5]
Writes outputs/24_day/retrieval/<fold>_<lap>.csv, retrieval_<fold>_<lap>.pdf and summary.md."""
import argparse
import os
import sys

sys.path.insert(0, '.')
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from odo import data, online
from odo.report import INK, INK2, SERIES

ap = argparse.ArgumentParser()
ap.add_argument('--fold', default='F5'); ap.add_argument('--lap', default='L2'); ap.add_argument('--every', type=int, default=5)
a = ap.parse_args()
out = 'outputs/24_day/retrieval'; os.makedirs(out, exist_ok=True)
cams = list(data.CAMS)
dt_ns = int(np.load(f'outputs/20_map/{cams[0]}.npz')['dt_ns'])
rel = online.Reloc(cams, data.fold_windows(a.fold), dt_ns=dt_ns)
sets = online.image_sets(a.lap, cams).iloc[::a.every].reset_index(drop=True)
roots = {'night': None, 'day': f'data/day_proxy/{a.lap}', 'day_nolights': f'data/day_proxy/{a.lap}_nolights'}
rows = []
gx, gy, _ = data.interp_gt(sets.stamp_ns.values + dt_ns)
for name, root in roots.items():
    if root and not os.path.exists(f'{root}/images/rr.csv'):
        continue
    for k in range(len(sets)):
        paths = [p.replace(data.DATA, os.path.abspath(root)) if root and isinstance(p, str) else p for p in sets.iloc[k][cams]]
        c = rel.candidates(paths)
        d = [np.hypot(p[0] - gx[k], p[1] - gy[k]) for p in c]
        rows.append(dict(images=name, t=(sets.stamp_ns[k] - data.t0_ns()) / 1e9, top1=d[0], best4=min(d)))
    print(name, 'done', flush=True)
r = pd.DataFrame(rows); r.to_csv(f'{out}/{a.fold}_{a.lap}.csv', index=False)
fig, axs = plt.subplots(2, 1, figsize=(11.69, 7), sharex=True)
for i, (name, g) in enumerate(r.groupby('images', sort=False)):
    axs[0].plot(g.t, g.top1.clip(upper=60), '.', ms=3, color=[INK2, SERIES[0], SERIES[1]][i], label=name)
    axs[1].plot(g.t, g.best4.clip(upper=60), '.', ms=3, color=[INK2, SERIES[0], SERIES[1]][i], label=name)
axs[0].set_ylabel('top-1 dist. to GT [m]'); axs[1].set_ylabel('best of 4 cand. [m]'); axs[1].set_xlabel('t [s]')
for ax in axs:
    ax.set_yscale('symlog', linthresh=5); ax.axhline(15, color=INK, lw=0.6, ls='--')
axs[0].legend(frameon=False, loc='upper left')
fig.suptitle(f'MegaLoc global retrieval vs night map ({a.fold}), {a.lap}: night vs day-proxy queries (clipped at 60 m; '
             'dashed = 15 m)', color=INK)
fig.tight_layout(); fig.savefig(f'{out}/retrieval_{a.fold}_{a.lap}.pdf'); plt.close(fig)
L = [f'# Global retrieval on day-proxy images, {a.fold} / {a.lap}', '',
     f'MegaLoc (odo.online.Reloc, 7-camera averaged similarity, 4 spatially distinct candidates) against the night map laps '
     f'of {a.fold}; every {a.every}th image set of {a.lap}. The online localiser verifies candidates with a wide-radius '
     'guided matching + PnP (radius 60 px), which succeeds when a candidate is within roughly 10-15 m of the truth.', '',
     '| query images | n | top-1 median [m] | top-1 <= 5 m | top-1 <= 15 m | best of 4 <= 15 m |', '|---|---|---|---|---|---|']
for name, g in r.groupby('images', sort=False):
    L.append(f'| {name} | {len(g)} | {g.top1.median():.1f} | {np.mean(g.top1 <= 5) * 100:.1f} % | '
             f'{np.mean(g.top1 <= 15) * 100:.1f} % | {np.mean(g.best4 <= 15) * 100:.1f} % |')
open(f'{out}/summary_{a.fold}_{a.lap}.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L), flush=True)
print('RETRIEVAL_DONE')
