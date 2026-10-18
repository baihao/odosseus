"""Per-frame error plot of one CV fold: columns = the fold's laps (validation laps, then the tuning lap), rows = position,
lateral, longitudinal and yaw error (GT car frame) vs time since the lap start. Grey strip under each position panel:
frames without an accepted fix (gated / no map / relocalisation).
    python scripts/plot_per_frame.py F5   -> outputs/21_online/cv/F5/per_frame_error.{png,pdf}, docs/figures/F5_per_frame_error.png"""
import sys, shutil; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from odo import data, report
fold = sys.argv[1]
laps = data.FOLDS[fold]['val'] + [data.tune_lap(fold)]
rows = [('pos', 'position error [m]'), ('lat', 'lateral error [m]'), ('lon', 'longitudinal error [m]'), ('dyaw', 'yaw error [°]')]
D = {}
for lap in laps:
    d = f'outputs/21_online/cv/{fold}/{lap}'
    m, per = report.compute_all(report.load_pred(f'{d}/pred.csv'), lap)
    inf = pd.read_csv(f'{d}/online_info.csv').set_index('timestamp_ns').reindex(per['ts'])
    t = (per['ts'] - per['ts'][0]) / 1e9
    D[lap] = (t, per, (inf.status != 'fix').values, m)
w = [D[l][0][-1] for l in laps]
fig, axs = plt.subplots(4, len(laps), figsize=(15, 9), sharey='row', gridspec_kw=dict(width_ratios=w), squeeze=False)
for j, lap in enumerate(laps):
    t, per, nofix, m = D[lap]
    role = 'tuning' if lap == data.tune_lap(fold) else 'validation'
    for i, (k, lab) in enumerate(rows):
        ax = axs[i, j]
        ax.plot(t, per[k], color=report.SERIES[0], lw=1.0)
        if k == 'pos':
            for th in (0.25, 1.0):
                ax.axhline(th, color=report.MUTED, lw=0.8, ls='--')
            ax.text(t[-1], 1.0, ' 1 m', va='center', ha='left', fontsize=7, color=report.INK2, clip_on=False) if j == len(laps) - 1 else None
            y0 = -0.06 * max(1.0, np.percentile(np.concatenate([D[l][1]['pos'] for l in laps]), 99.5))
            ax.fill_between(t, y0, 0, where=nofix, color=report.MUTED, step='mid', lw=0)
            ax.set_title(f'{lap} ({role})   RMSE {m["pos_rmse"]:.3f} m, max {per["pos"].max():.2f} m', fontsize=9, color=report.INK, loc='left')
        else:
            ax.axhline(0, color=report.MUTED, lw=0.8)
        if j == 0:
            ax.set_ylabel(lab)
        if i == 3:
            ax.set_xlabel('time since lap start [s]')
        ax.margins(x=0)
fig.text(0.01, 0.005, f'{fold}: map laps {" ".join(data.FOLDS[fold]["map_laps"])}. Grey strip below 0 in the position row: frames '
         'without an accepted map fix (dead reckoning only). Dashed: 0.25 m and 1 m. Errors in the GT car frame.', fontsize=8, color=report.INK2)
fig.suptitle(f'Exp21, fold {fold}: per-frame error', x=0.01, ha='left', fontsize=12, color=report.INK, weight='bold')
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
out = f'outputs/21_online/cv/{fold}/per_frame_error'
fig.savefig(out + '.png', dpi=110); fig.savefig(out + '.pdf')
shutil.copy(out + '.png', f'docs/figures/{fold}_per_frame_error.png')
print('saved', out)
