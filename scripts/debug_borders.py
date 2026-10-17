import sys; sys.path.insert(0, '.')
import numpy as np, cv2
from scipy.spatial.transform import Rotation
from odo import data, borders
from odo.online import Cams, image_sets
from odo.reloc_pnp import GT6
from odo.map_build import DT_NS
from odo.prep import CROP_BOTTOM
cams = data.CAMS; cm = Cams(cams); bd = borders.Borders(); gt6 = GT6()
ycut = [CROP_BOTTOM[c] * cm.H for c in cams]
sets = image_sets('tune', cams)
for k in [100, 400, 700]:
    st = int(sets.stamp_ns[k]); T = gt6.T(st + DT_NS)[0]
    yaw = np.arctan2(T[1, 0], T[0, 0]); R_rp = Rotation.from_euler('z', -yaw).as_matrix() @ T[:3, :3]
    Xg = np.array([T[0, 3], T[1, 3], yaw]); c, s = np.cos(yaw), np.sin(yaw)
    X0 = Xg + np.array([-s * 0.5, c * 0.5, 0])
    ims = [cv2.imread(p, 0) for p in sets.iloc[k][cams]]
    g = borders.edge_maps(ims)
    H, d, stg, q, ci = borders.border_measure(bd, cm, g, X0, T[2, 3], R_rp, [True] * 7, ycut=ycut)
    pred = -H @ (X0 - Xg)
    for i, cn in enumerate(cams):
        m = ci == i
        if m.sum() < 3: continue
        good = np.abs(d[m] - pred[m]) < 2
        print(k, cn, 'n', m.sum(), 'expected offset med %.1f' % np.median(pred[m]), 'measured med %.1f' % np.median(d[m]), 'frac consistent %.2f' % good.mean(), 'q med %.1f' % np.median(q[m]))
