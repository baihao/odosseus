"""Stage 3: track-border edge alignment (method 2 of the brief), an appearance-independent cue.

The surveyed 3D track borders (track_bounds: left/right track and pit-lane borders) are projected into every camera
with the predicted pose. At sample points along each projected border, the image is searched along the normal of the
projected curve for the strongest intensity edge (directional Sobel derivative, |.|, within +-`search_px`, weighted
by a Gaussian prior around the prediction). Each sample gives a scalar measurement "edge offset along the normal"
whose Jacobian w.r.t. (x, y, yaw) comes from the projection. They are fused as scalar EKF measurements in information
form with IRLS (Cauchy) weights; on straights they constrain lateral position and yaw only, in corners also the
position along the track.

    python -m odo.borders --split tune --cams fl fr cl cr r rl rr   # offline check at GT pose (bias / spread / recovery)
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from . import data

DEV = 'cuda'


class Borders:
    def __init__(self, step=0.5, keys=('track_left', 'track_right', 'pitlane_left', 'pitlane_right')):
        self.lines = []
        for k in keys:
            P = data.track_bounds()[k]
            seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
            s = np.concatenate([[0], np.cumsum(seg)])
            ss = np.arange(0, s[-1], step)
            Q = np.stack([np.interp(ss, s, P[:, j]) for j in range(3)], 1)
            self.lines.append(Q)
        self.P = np.concatenate(self.lines)
        self.line_id = np.concatenate([np.full(len(l), i) for i, l in enumerate(self.lines)])
        # tangent (3D) per point, from neighbours on the same line
        T = np.zeros_like(self.P); o = 0
        for l in self.lines:
            t = np.gradient(l, axis=0); T[o:o + len(l)] = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9); o += len(l)
        self.T = T
        self.tree = cKDTree(self.P[:, :2])

    def near(self, x, y, r=45.0):
        return np.array(self.tree.query_ball_point([x, y], r), dtype=int)


def edge_maps(ims, scale=0.5):
    """Grayscale uint8 list -> (B,2,h,w) Sobel gradients (float, at `scale`), torch on GPU."""
    x = torch.from_numpy(np.stack(ims)).to(DEV).float()[:, None] / 255.0
    if scale != 1:
        x = F.interpolate(x, scale_factor=scale, mode='area')
    x = F.avg_pool2d(x, 3, 1, 1)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=DEV) / 8
    g = F.conv2d(x, torch.stack([kx, kx.T])[:, None], padding=1)
    return g


def project_pts(cm, X3, ci, Rwb, twb):
    """numpy projection (reuses online.project_np camera model)."""
    from .online import project_np
    return project_np(cm, X3, ci, Rwb, twb)


def border_measure(bd, cm, grads, X, z, R_rp, cam_ok, scale=0.5, search_px=20, prior_px=10, min_d=4.0, max_d=40.0,
                   ycut=None):
    """Returns per-sample (H (n,3), d (n,), q (n,) edge strength, cam idx (n,)) around prior pose X."""
    idx = bd.near(X[0], X[1], max_d + 5)
    if len(idx) == 0:
        return None
    P3, T3 = bd.P[idx], bd.T[idx]
    ncam = len(cm.names)
    c, s = np.cos(X[2]), np.sin(X[2])
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    Rwb = Rz @ R_rp
    twb = np.array([X[0], X[1], z])
    # camera centres to pre-filter by distance
    out = []
    eps = np.array([1e-3, 1e-3, 1e-5])
    for i in range(ncam):
        if not cam_ok[i]:
            continue
        Tcb_R, Tcb_t = cm.Rcb_np[i], cm.tcb_np[i]
        Xc = (P3 - twb) @ Rwb @ Tcb_R.T + Tcb_t
        sel = (Xc[:, 2] > min_d) & (np.linalg.norm(Xc, axis=1) < max_d)
        if sel.sum() < 5:
            continue
        Ps, Ts = P3[sel], T3[sel]
        ci = np.full(len(Ps), i)
        # projections at X and perturbed poses (for the Jacobian), plus a point along the tangent (for the normal)
        Ps_t = np.concatenate([Ps, Ps + 0.2 * Ts])
        ci2 = np.full(len(Ps_t), i)
        PP = []
        for k, dp in enumerate([np.zeros(3)] + list(np.diag(eps))):
            Xp = X + dp
            cc, ss_ = np.cos(Xp[2]), np.sin(Xp[2])
            Rp = np.array([[cc, -ss_, 0], [ss_, cc, 0], [0, 0, 1]]) @ R_rp
            uv, ok = project_pts(cm, Ps_t if k == 0 else Ps, ci2 if k == 0 else ci, Rp[None], np.array([[Xp[0], Xp[1], z]]))
            PP.append((uv[0], ok[0]))
        uv0, ok0 = PP[0][0][:len(Ps)], PP[0][1][:len(Ps)] & PP[0][1][len(Ps):]
        tan = PP[0][0][len(Ps):] - uv0
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        J = np.stack([(PP[k + 1][0] - uv0) / eps[k] for k in range(3)], 2)        # (n,2,3)
        H = np.einsum('nk,nkj->nj', nrm, J)
        inside = ok0 & (uv0[:, 0] > 5) & (uv0[:, 0] < cm.W - 5) & (uv0[:, 1] > 5) & (uv0[:, 1] < (ycut[i] if ycut else cm.H) - 5)
        if inside.sum() < 5:
            continue
        uv0, nrm, H = uv0[inside], nrm[inside], H[inside]
        # sample the directional derivative along the normal
        ts = torch.arange(-search_px, search_px + 1, 1.0, device=DEV)
        u = torch.from_numpy(uv0).float().to(DEV); nn_ = torch.from_numpy(nrm).float().to(DEV)
        pts = (u[:, None] + ts[None, :, None] * nn_[:, None]) * scale                # (n,L,2) at edge-map scale
        g = grads[i:i + 1]
        h, w = g.shape[-2:]
        grid = torch.stack([pts[..., 0] / (w - 1) * 2 - 1, pts[..., 1] / (h - 1) * 2 - 1], -1)[None]
        gs = F.grid_sample(g, grid, align_corners=True)[0]                          # (2,n,L)
        dd = (gs[0] * nn_[:, None, 0] + gs[1] * nn_[:, None, 1]).abs()
        wprior = torch.exp(-0.5 * (ts / prior_px) ** 2)
        score = dd * wprior[None]
        best = score.argmax(1)
        tb = ts[best]
        strength = dd.gather(1, best[:, None])[:, 0]
        med = dd.median(1).values
        q = (strength / (med + 1e-3)).cpu().numpy()
        out.append((H, tb.cpu().numpy(), strength.cpu().numpy(), q, np.full(len(H), i)))
    if not out:
        return None
    return [np.concatenate([o[k] for o in out]) for k in range(5)]


def border_update(X, P, meas, sigma_px=3.0, n_eff=15, min_strength=0.02, min_q=2.0, iters=4, cauchy=2.0, X_lin=None):
    """Robust (IRLS, Cauchy) information-form update with scalar edge-offset measurements linearised at X_lin
    (iterated EKF; default X_lin = prior mean X). Returns X, P, info."""
    X_lin = X if X_lin is None else X_lin
    H, d, st, q, ci = meas
    keep = (st > min_strength) & (q > min_q)
    H, d, ci = H[keep], d[keep], ci[keep]
    if len(d) < 10:
        return X, P, dict(b_n=int(len(d)))
    # correlated samples: inflate sigma so that each camera contributes at most ~n_eff independent samples
    sig = np.full(len(d), sigma_px)
    for i in np.unique(ci):
        m = ci == i
        sig[m] *= np.sqrt(max(m.sum() / n_eff, 1.0))
    Pinv = np.linalg.inv(P)
    dprior = X - X_lin; dprior[2] = (dprior[2] + np.pi) % (2 * np.pi) - np.pi
    w = np.ones(len(d)); dx = np.zeros(3)
    for _ in range(iters):
        A = Pinv + (H.T * (w / sig ** 2)) @ H
        b = (H.T * (w / sig ** 2)) @ d + Pinv @ dprior
        dx = np.linalg.solve(A, b)
        r = d - H @ dx
        w = 1.0 / (1.0 + (r / (cauchy * sigma_px)) ** 2)
    Pn = np.linalg.inv(A)
    Xn = X_lin + dx
    return Xn, Pn, dict(b_n=int(len(d)), b_inl=int((np.abs(r) < 2 * sigma_px).sum()), b_dx=float(np.hypot(dx[0], dx[1])))


def check(split='tune', cams=tuple(data.CAMS), step_s=1.0, perturb=(0.0, 0.5, 0.0), out=None):
    """At GT pose (+ perturbation in car frame lon/lat [m], yaw [deg]): edge-offset statistics per camera and the
    pose after one border update with a broad prior."""
    from scipy.spatial.transform import Rotation
    from .online import Cams, image_sets
    from .reloc_pnp import GT6
    from .map_build import DT_NS
    cams = list(cams)
    cm = Cams(cams); bd = Borders(); gt6 = GT6()
    from .prep import CROP_BOTTOM
    ycut = [CROP_BOTTOM[c] * cm.H for c in cams]
    sets = image_sets(split, cams)
    import cv2
    res, stats = [], {c: [] for c in cams}
    for k in range(0, len(sets), int(step_s * 10)):
        st = int(sets.stamp_ns[k]); T = gt6.T(st + DT_NS)[0]
        yaw = np.arctan2(T[1, 0], T[0, 0]); R_rp = Rotation.from_euler('z', -yaw).as_matrix() @ T[:3, :3]
        Xg = np.array([T[0, 3], T[1, 3], yaw])
        c, s = np.cos(yaw), np.sin(yaw)
        X0 = Xg + np.array([c * perturb[0] - s * perturb[1], s * perturb[0] + c * perturb[1], np.radians(perturb[2])])
        ims = [cv2.imread(p, 0) if p else np.zeros((cm.H, cm.W), np.uint8) for p in sets.iloc[k][cams]]
        g = edge_maps(ims)
        P0 = np.diag([1.0, 1.0, np.radians(1.0)]) ** 2
        mg = border_measure(bd, cm, g, Xg, T[2, 3], R_rp, [True] * len(cams), ycut=ycut)
        if mg is not None:
            for i, cname in enumerate(cams):
                m = (mg[4] == i) & (mg[3] > 2.0)
                if m.sum():
                    stats[cname].append(np.median(mg[1][m]))
        mm = border_measure(bd, cm, g, X0, T[2, 3], R_rp, [True] * len(cams), ycut=ycut)
        if mm is None:
            continue
        Xn, Pn, info = border_update(X0, P0, mm)
        for _ in range(2):  # iterated EKF: re-linearise at the new estimate
            mm = border_measure(bd, cm, g, Xn, T[2, 3], R_rp, [True] * len(cams), ycut=ycut)
            if mm is None:
                break
            Xn, Pn, info = border_update(X0, P0, mm, X_lin=Xn)
        e0 = X0 - Xg; e1 = Xn - Xg
        lat = lambda e: -s * e[0] + c * e[1]
        lon = lambda e: c * e[0] + s * e[1]
        res.append((st, lat(e0), lat(e1), lon(e1), np.degrees(e1[2]), info.get('b_n', 0)))
    res = np.array(res)
    print('per-camera median edge offset at GT pose [px]:', {c: (round(float(np.median(v)), 2), round(float(np.std(v)), 2)) for c, v in stats.items() if v})
    print(f'lat err before {np.sqrt(np.mean(res[:, 1] ** 2)):.3f} after: rmse {np.sqrt(np.mean(res[:, 2] ** 2)):.3f} '
          f'median |.| {np.median(np.abs(res[:, 2])):.3f} | lon rmse {np.sqrt(np.mean(res[:, 3] ** 2)):.3f} | yaw rmse {np.sqrt(np.mean(res[:, 4] ** 2)):.3f} deg | n {np.median(res[:, 5])}')
    if out:
        np.save(out, res)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='tune'); ap.add_argument('--cams', nargs='+', default=data.CAMS)
    ap.add_argument('--perturb', nargs=3, type=float, default=[0.0, 0.5, 0.0]); ap.add_argument('--step', type=float, default=1.0)
    a = ap.parse_args()
    check(a.split, a.cams, a.step, a.perturb)
