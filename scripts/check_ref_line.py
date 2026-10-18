"""Diagnostic: track borders, pit lane, night lap GT paths (L3, L6, L7), the L9 estimate and the F5/L2 night
estimate in one map, with zooms; signed offset of each line from the track centre line.
    python scripts/check_ref_line.py -> outputs/24_day/L9/final/prafful_viz/ref_line_check.png"""
import json
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from odo import data

b = {k: np.asarray(v)[:, :2] for k, v in json.load(open(f'{data.DATA}/track_bounds.json')).items()}
g = data.gt()
laps = {}
for l in ('L3', 'L6', 'L7', 'L2'):
    a, e = data.LAPS[l]
    m = (g.timestamp_ns >= data.t0_ns() + int(a * 1e9)) & (g.timestamp_ns < data.t0_ns() + int(e * 1e9))
    laps[l] = g[m][['x', 'y']].values
l9 = pd.read_csv('outputs/24_day/L9/final/pred.csv')[['x', 'y']].values
n2 = pd.read_csv('outputs/24_day/night/final/F5_L2/pred.csv')[['x', 'y']].values
for k, v in b.items():
    print(k, len(v), 'pts, extent x', v[:, 0].min().round(1), v[:, 0].max().round(1), 'y', v[:, 1].min().round(1), v[:, 1].max().round(1))
for k, v in laps.items():
    print(k, 'GT extent x', v[:, 0].min().round(1), v[:, 0].max().round(1), 'y', v[:, 1].min().round(1), v[:, 1].max().round(1))
print('L9 est extent x', l9[:, 0].min().round(1), l9[:, 0].max().round(1), 'y', l9[:, 1].min().round(1), l9[:, 1].max().round(1))

from scipy.spatial import cKDTree
tl, tr = cKDTree(b['track_left']), cKDTree(b['track_right'])
for name, p in [('L7 GT', laps['L7']), ('L3 GT', laps['L3']), ('L9 est', l9), ('F5/L2 est', n2)]:
    dl, _ = tl.query(p); dr, _ = tr.query(p)
    w = dl + dr
    print(f'{name}: dist to left border median {np.median(dl):.1f} m, to right {np.median(dr):.1f} m, '
          f'width-sum p5/50/95 {np.percentile(w, 5):.1f}/{np.median(w):.1f}/{np.percentile(w, 95):.1f} m, '
          f'frac with sum > 25 m (outside?) {np.mean(w > 25) * 100:.1f} %')

fig, axs = plt.subplots(1, 3, figsize=(24, 9))
zooms = [None, ((-200, 0), (550, 750)), ((150, 350), (-50, 150))]
for ax, z in zip(axs, zooms):
    for k, v in b.items():
        ax.plot(v[:, 0], v[:, 1], color='#bbb' if 'pit' in k else '#666', lw=0.8)
    for l, c in (('L3', 'tab:orange'), ('L6', 'tab:green'), ('L7', 'tab:red')):
        ax.plot(laps[l][:, 0], laps[l][:, 1], color=c, lw=0.8, label=f'{l} GT')
    ax.plot(l9[:, 0], l9[:, 1], color='tab:blue', lw=1.2, label='L9 estimate')
    ax.set_aspect('equal')
    if z:
        ax.set_xlim(*z[0]); ax.set_ylim(*z[1])
axs[0].legend()
fig.tight_layout(); fig.savefig('outputs/24_day/L9/final/prafful_viz/ref_line_check.png', dpi=80)
print('CHECK_DONE')
