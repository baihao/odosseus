"""Border cue sensitivity (tune lap, every 2 s): pixel shift of the projected borders for a 0.5 m lateral pose shift,
and the fraction of edge samples whose detected edge is consistent (+-2 px) with the true border position."""
import sys; sys.path.insert(0, '.')
import numpy as np, cv2, pandas as pd
from scipy.spatial.transform import Rotation
from odo import data, borders
from odo.online import Cams, image_sets
from odo.reloc_pnp import GT6
from odo.map_build import DT_NS
from odo.prep import CROP_BOTTOM
cams = data.CAMS; cm = Cams(cams); bd = borders.Borders(); gt6 = GT6()
ycut = [CROP_BOTTOM[c] * cm.H for c in cams]
sets = image_sets('tune', cams)
rows = []
for k in range(0, len(sets), 20):
    st = int(sets.stamp_ns[k]); T = gt6.T(st + DT_NS)[0]
    yaw = np.arctan2(T[1, 0], T[0, 0]); R_rp = Rotation.from_euler('z', -yaw).as_matrix() @ T[:3, :3]
    Xg = np.array([T[0, 3], T[1, 3], yaw]); c, s = np.cos(yaw), np.sin(yaw)
    X0 = Xg + np.array([-s * 0.5, c * 0.5, 0])
    g = borders.edge_maps([cv2.imread(p, 0) for p in sets.iloc[k][cams]])
    mm = borders.border_measure(bd, cm, g, X0, T[2, 3], R_rp, [True] * 7, ycut=ycut)
    if mm is None: continue
    H, d, stg, q, ci = mm
    pred = -H @ (X0 - Xg)
    for i, cn in enumerate(cams):
        m = ci == i
        if m.sum() < 3: continue
        rows.append(dict(cam=cn, shift_med=np.median(np.abs(pred[m])), shift_p90=np.percentile(np.abs(pred[m]), 90),
                         consistent=np.mean(np.abs(d[m] - pred[m]) < 2), n=m.sum()))
r = pd.DataFrame(rows).groupby('cam').agg(frames=('n', 'size'), samples=('n', 'median'), shift_med_px=('shift_med', 'median'),
                                          shift_p90_px=('shift_p90', 'median'), consistent_frac=('consistent', 'mean')).round(2)
print(r.to_string()); r.to_csv('outputs/22_borders/sensitivity.csv')
