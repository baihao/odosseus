"""Turn a camera VO trajectory (arbitrary world frame, arbitrary/drifting scale) into map-frame
base_link poses x, y, yaw.

Per consecutive frame pair:
  T_c(i-1)c(i) from VO  ->  rotation to base frame with T_base_cam  ->  yaw increment
  translation direction from VO, metric step length from one of the scale sources:
    'speed'  : integrated car-state speed between the two stamps (allowed input, causal)
    'heading': rotation from VO only; step of speed length along the chord of the heading (beta = 0)
    'vo'     : VO's own translation times one global scale fitted to GT (diagnostic only, not causal)
  then compose from the first pose (allowed: GT at the first stamp of the segment).
Frames flagged invalid (tracking lost) keep the previous step's yaw rate and use speed for length.
"""
import numpy as np

from . import data


def quat_to_R(q):  # x y z w
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def speed_dist(ts_ns):
    """Distance travelled between consecutive stamps from integrated car-state speed."""
    s = data.allowed_state()
    tt = s.timestamp_ns.values.astype(np.float64)
    v = s.v_mps.values
    cum = np.concatenate([[0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(tt) / 1e9)])
    c = np.interp(np.asarray(ts_ns, dtype=np.float64), tt, cum)
    return np.diff(c)


def rel_steps(T_wc, T_bc):
    """T_wc: (N,4,4) camera-to-world (rotation orthonormal, translation in VO units).
    Returns per step (N-1): base-frame rotation (3x3), camera-frame translation (3) in VO units."""
    Rs, ts = [], []
    for a, b in zip(T_wc[:-1], T_wc[1:]):
        Rrel = a[:3, :3].T @ b[:3, :3]
        trel = a[:3, :3].T @ (b[:3, 3] - a[:3, 3])
        Rs.append(Rrel); ts.append(trel)
    return np.array(Rs), np.array(ts)


def to_map(ts_ns, T_wc, cam, scale='speed', valid=None, vo_scale=1.0, x0=None):
    """Returns x, y, yaw arrays (len N) in map frame."""
    T_bc = data.calib()[cam]['T_base_cam']
    Rbc, tbc = T_bc[:3, :3], T_bc[:3, 3]
    N = len(ts_ns)
    valid = np.ones(N, bool) if valid is None else np.asarray(valid, bool)
    Rrel, trel = rel_steps(T_wc, T_bc)
    dist = speed_dist(ts_ns)
    x, y, th = np.zeros(N), np.zeros(N), np.zeros(N)
    x[0], y[0], th[0] = data.first_pose(ts_ns[0]) if x0 is None else x0
    last_dyaw = 0.0
    for i in range(1, N):
        ok = valid[i] and valid[i - 1]
        if ok:
            Rb = Rbc @ Rrel[i - 1] @ Rbc.T
            dyaw = np.arctan2(Rb[1, 0], Rb[0, 0])
            d_cam = trel[i - 1]
            if scale in ('speed', 'heading'):
                db = Rbc @ d_cam
                n = np.linalg.norm(db[:2])
                k = dist[i - 1] / n if n > 1e-9 else 0.0
            else:
                k = vo_scale
            # metric base displacement including the camera lever arm: t_b = Rbc t_c k + tbc - Rb tbc
            tb = Rbc @ (d_cam * k) + tbc - Rb @ tbc
            if scale == 'heading' or (scale == 'speed' and np.linalg.norm(Rbc @ d_cam) < 1e-9):
                # translation along the chord of the rotated heading, length from speed
                tb = dist[i - 1] * np.array([np.cos(dyaw / 2), np.sin(dyaw / 2), 0])
            last_dyaw = dyaw
        else:
            dyaw = last_dyaw
            tb = np.array([dist[i - 1], 0, 0])
        c, s = np.cos(th[i - 1]), np.sin(th[i - 1])
        x[i] = x[i - 1] + c * tb[0] - s * tb[1]
        y[i] = y[i - 1] + s * tb[0] + c * tb[1]
        th[i] = th[i - 1] + dyaw
    return x, y, np.angle(np.exp(1j * th))


def fit_vo_scale(ts_ns, T_wc, cam):
    """Diagnostic: single global scale making VO step lengths match GT step lengths (median ratio)."""
    gx, gy, _ = data.interp_gt(ts_ns)
    g = np.hypot(np.diff(gx), np.diff(gy))
    v = np.linalg.norm(np.diff(T_wc[:, :3, 3], axis=0), axis=1)
    m = (g > 0.5) & (v > 0)
    return float(np.median(g[m] / v[m])) if m.any() else 1.0


def sim3_rows_to_T(rows):
    """rows: (N, 8) t(3) q(4: x y z w) s -> (N,4,4) with pure rotation, translation as is
    (Sim3 scale only scales the local point cloud; camera centre is t)."""
    T = np.tile(np.eye(4), (len(rows), 1, 1))
    for i, r in enumerate(rows):
        T[i, :3, :3] = quat_to_R(r[3:7]); T[i, :3, 3] = r[:3]
    return T
