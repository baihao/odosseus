"""Diagnostic: signed offset of the L9 estimate from the L7 GT line, Line.xy_to_sn (prafful viz adapter) vs a plain
nearest-point distance; where they disagree."""
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, '.')
sys.argv = sys.argv[:1]
import importlib.util
spec = importlib.util.spec_from_file_location('viz', 'scripts/exp24_prafful_viz.py')
viz = importlib.util.module_from_spec(spec); spec.loader.exec_module(viz)

lx, ly = viz.lap_line('L7')
line = viz.Line(lx, ly)
p = pd.read_csv('outputs/24_day/L9/final/pred.csv')
s, n, _, _ = line.xy_to_sn(p.x.values, p.y.values)
dense = cKDTree(np.c_[lx, ly])
dd, _ = dense.query(np.c_[p.x.values, p.y.values])
print('Line n: |n| p50/p95/max', np.percentile(np.abs(n), [50, 95, 100]).round(2))
print('nearest dist to raw L7 GT points: p50/p95/max', np.percentile(dd, [50, 95, 100]).round(2))
seg = np.hypot(np.diff(lx), np.diff(ly))
print('L7 GT: points', len(lx), 'step p50/max', np.median(seg).round(3), seg.max().round(2), 'line length', seg.sum().round(1))
print('tangent norm min', np.linalg.norm(np.gradient(line.p, axis=0), axis=1).min())
bad = np.abs(np.abs(n) - dd) > 0.5
print('frames where |n| differs from nearest distance by > 0.5 m:', bad.sum(), 'of', len(n))
print('OFFSET_DONE')
