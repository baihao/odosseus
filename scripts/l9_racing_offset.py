"""Offset of the L9 estimate from the organizers' racing line (track-frame n, + = left), for the L9 runs; the night
laps' GT for context (how far a real lap drives from the line).
    python scripts/l9_racing_offset.py -> outputs/24_day/L9/racing_line_offset.md"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
sys.path.insert(0, 'data/23-00-38_cut_120m_update/tools')
from track_frame import load_lines
from odo import data

line = load_lines('data/23-00-38_cut_120m_update/map.mcap')['track']


def stats(name, t, x, y, skip_s=None):
    t = np.asarray(t, np.int64)
    _, n, _, _ = line.xy_to_sn(x, y, t_ns=t)
    rows = [(name, 'whole run', n)]
    if skip_s:
        rows.append((name, f'after the first {skip_s:g} s', n[(t - t[0]) / 1e9 >= skip_s]))
    return [dict(run=r, part=p, frames=len(v), mean_signed=v.mean(), mean_abs=np.abs(v).mean(), median_abs=np.median(np.abs(v)),
                 rms=np.sqrt(np.mean(v ** 2)), p95_abs=np.percentile(np.abs(v), 95), max_abs=np.abs(v).max()) for r, p, v in rows]


R = []
for run, label in (('final', 'L9 estimate, map = all night laps L0–L8'), ('F5', 'L9 estimate, map = F5 (L0 L1 L3 L4 L6 L8)')):
    p = pd.read_csv(f'outputs/24_day/L9/{run}/pred.csv')
    R += stats(label, p.timestamp_ns.values, p.x.values, p.y.values, skip_s=10)
g = data.gt()
for lap in ('L3', 'L6', 'L7'):
    a, b = data.LAPS[lap]
    m = (g.timestamp_ns >= data.t0_ns() + int(a * 1e9)) & (g.timestamp_ns < data.t0_ns() + int(b * 1e9))
    gg = g[m].iloc[::10]                                       # 10 Hz like the estimates
    R += stats(f'night {lap} GT (flying lap)', gg.timestamp_ns.values, gg.x.values, gg.y.values)
df = pd.DataFrame(R)
L = ['# L9: offset from the organizers\' racing line', '',
     'n of the organizers\' track frame (`data/23-00-38_cut_120m_update`, `tools/track_frame.py`), + = left of the line, '
     'metres. No ground truth for L9: this is how close the estimate stays to the line, not an error. Night-lap GT rows '
     'show how far real laps drive from the line.', '',
     '| run | part | frames | mean (signed) | mean abs | median abs | RMS | p95 abs | max abs |', '|---|---|---|---|---|---|---|---|---|']
for r in R:
    L.append(f'| {r["run"]} | {r["part"]} | {r["frames"]} | {r["mean_signed"]:+.3f} | {r["mean_abs"]:.3f} | '
             f'{r["median_abs"]:.3f} | {r["rms"]:.3f} | {r["p95_abs"]:.3f} | {r["max_abs"]:.3f} |')
open('outputs/24_day/L9/racing_line_offset.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
print('OFFSET_DONE')
