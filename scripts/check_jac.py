import sys; sys.path.insert(0, '.')
import numpy as np
from scipy.spatial.transform import Rotation
from odo import data, online
cm = online.Cams(data.CAMS)
rng = np.random.default_rng(0)
n = 50; ci = rng.integers(0, 7, n)
p = np.array([10.0, 5.0, 0.3, 0.01, -0.02, 0.05]); z = 1.0
R_rp = Rotation.from_euler('xy', [0.01, 0.02]).as_matrix()
# points in front of each camera
Rwb = online.pose_R(p, R_rp); t = np.array([p[0], p[1], z + p[5]])
Xc = np.stack([rng.uniform(-3, 3, n), rng.uniform(-1, 1, n), rng.uniform(5, 30, n)], 1)
Tbc = np.linalg.inv(np.stack([np.eye(4)] * 7)); Xw = []
for k in range(n):
    Tcb_R, Tcb_t = cm.Rcb_np[ci[k]], cm.tcb_np[ci[k]]
    Xb = Tcb_R.T @ (Xc[k] - Tcb_t); Xw.append(Rwb @ Xb + t)
Xw = np.array(Xw); uv = rng.normal(0, 1, (n, 2)) + 500
r, ok, J = online.pnp_lin(cm, Xw, ci, uv, p, z, R_rp)
eps = np.array([1e-6, 1e-6, 1e-8, 1e-8, 1e-8, 1e-6])
Jn = np.stack([(online.pnp_lin(cm, Xw, ci, uv, p + np.eye(6)[k] * eps[k], z, R_rp, jac=False)[0] - r) / eps[k] for k in range(6)], 2)
print('ok frac', ok.mean(), 'max rel err', np.abs(J - Jn).max() / np.abs(Jn).max(), 'per-param', (np.abs(J - Jn).max((0, 1)) / np.abs(Jn).max((0, 1))).round(6))
