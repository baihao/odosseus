"""Lateral vs longitudinal (its complement: along-track) error of one CV lap, shared time axis and y-scale, plus the share of
the squared position error carried by each component.
    python scripts/plot_lat_lon.py F5 L2 -> outputs/21_online/cv/F5/L2/lat_lon.{png,pdf}, docs/figures/F5_L2_lat_lon.png"""
import sys, shutil; sys.path.insert(0, '.')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from odo import data, report
fold, lap = sys.argv[1], sys.argv[2]
d = f'outputs/21_online/cv/{fold}/{lap}'
m, per = report.compute_all(report.load_pred(f'{d}/pred.csv'), lap)
t = (per['ts'] - data.t0_ns()) / 1e9
lat, lon, pos = per['lat'], per['lon'], per['pos']
sh_lat = np.sum(lat ** 2) / np.sum(pos ** 2) * 100
fig, axs = plt.subplots(3, 1, figsize=(13, 7), sharex=True, sharey=True)
for a, v, lab, title in [(axs[0], pos, 'position error [m]', f'position error  (RMSE {m["pos_rmse"]:.3f} m)'),
                         (axs[1], lat, 'lateral error [m]', f'lateral error, across the GT heading  (RMSE {m["lat_rmse"]:.3f} m, {sh_lat:.0f} % of squared error)'),
                         (axs[2], lon, 'longitudinal error [m]', f'longitudinal error, along the GT heading  (RMSE {m["lon_rmse"]:.3f} m, {100 - sh_lat:.0f} % of squared error)')]:
    a.plot(t, v, color=report.SERIES[0], lw=1)
    a.axhline(0, color=report.MUTED, lw=.8)
    a.set_ylabel(lab); a.set_title(title, loc='left', fontsize=9, color=report.INK); a.margins(x=0)
axs[2].set_xlabel('run time t [s]')
fig.suptitle(f'Exp21 {fold} / {lap}: lateral and longitudinal error', x=0.01, ha='left', fontsize=12, weight='bold', color=report.INK)
fig.text(0.01, 0.005, 'GT car frame: lateral > 0 = estimate left of the GT, longitudinal > 0 = estimate ahead. '
         'lateral² + longitudinal² = position². Same y-scale in all rows.', fontsize=8, color=report.INK2)
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
out = f'{d}/lat_lon'; fig.savefig(out + '.png', dpi=110); fig.savefig(out + '.pdf')
shutil.copy(out + '.png', f'docs/figures/{fold}_{lap}_lat_lon.png')
pk = np.argmax(pos)
print(f'lat RMSE {m["lat_rmse"]:.3f} lon RMSE {m["lon_rmse"]:.3f}; lateral share of squared error {sh_lat:.0f} %; '
      f'at peak t={t[pk]:.1f}: lat {lat[pk]:+.2f} lon {lon[pk]:+.2f}; max |lon| {np.abs(lon).max():.2f} at t={t[np.argmax(np.abs(lon))]:.1f}')
