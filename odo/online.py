"""Stage 4: causal real-time localiser = car-state dead reckoning (EKF prediction) + map-based 3-DoF PnP fixes.

Per image set (7 cameras, one trigger stamp t):
  1. EKF predict from the previous image set to t with the allowed car-state signals (vx, vy, wz; only samples
     with stamp <= t are used).
  2. Pick map frames (cached 3D feature map of the training run, odo.map_build) near the predicted pose: the nearest
     frame of each of the `n_pass` closest passes (laps) with a similar heading.
  3. SuperPoint on all cameras in one batch. Guided matching: map 3D points are projected with the predicted pose
     (plumb_bob distortion) and matched to query keypoints within a radius by descriptor similarity (mutual best).
     A second pass re-projects with the refined pose and a small radius.
  4. 3-DoF (x, y, yaw) robust Gauss-Newton over all cameras; z/roll/pitch from the nearest map frame.
  5. Gated EKF update (Mahalanobis); repeated rejections of strong fixes reset the filter.
The pose is output at the image stamp (state is kept at the GT-clock time t + dt, see Exp12) for every image set,
fix or not, so coverage is always 100 %.

    python -m odo.online --split val --out outputs/21_online/val
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from . import data, report
from .prep import CROP_BOTTOM

DEV = 'cuda'


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------ map
class FeatureMap:
    """Cached map, restricted to map frames whose own and partner stamps lie in `windows` (list of (t0, t1) ns)."""

    def __init__(self, map_dir, cams, windows, max_gt_acc=1.0, risk_margin_s=3.0):
        self.cams = {}
        self.max_gt_acc = max_gt_acc
        inw = lambda s: np.any([(s >= a) & (s < b) for a, b in windows], axis=0)
        # training-GT quality: the state estimator's own position accuracy. risk(frame) = max accuracy value within
        # +-risk_margin_s (its error outlasts the flag). Frames whose own GT is flagged are dropped; risky frames are
        # used only when no safe pass is near (see select()).
        sd = data.state()
        acc = np.hypot(sd['pos_accuracy.0'].values, sd['pos_accuracy.1'].values)
        ts_ = sd.timestamp_ns.values
        gt_acc = lambda s: np.interp(s.astype(np.float64), ts_.astype(np.float64), acc)
        k = int(risk_margin_s * 100)
        from scipy.ndimage import maximum_filter1d
        acc_max = maximum_filter1d(acc, 2 * k + 1)
        risk = lambda s: acc_max[np.clip(np.searchsorted(ts_, s), 0, len(ts_) - 1)]
        for cam in cams:
            z = np.load(f'{map_dir}/{cam}.npz')
            st, sp, sn = z['stamp'], z['stamp_prev'], z['stamp_next']
            keep = inw(st) & ((sp == 0) | inw(sp)) & ((sn == 0) | inw(sn)) & (z['count'] > 0)
            for ss in (st, sp, sn):
                keep &= (ss == 0) | (gt_acc(ss) < max_gt_acc)
            off, cnt = z['offset'][keep], z['count'][keep]
            idx = np.concatenate([np.arange(o, o + c) for o, c in zip(off, cnt)])
            T = z['T'][keep]
            self.cams[cam] = dict(
                stamp=st[keep], T=T, xy=T[:, :2, 3].copy(), yaw=np.arctan2(T[:, 1, 0], T[:, 0, 0]), risk=risk(st[keep]),
                offset=torch.from_numpy(np.concatenate([[0], np.cumsum(cnt)[:-1]])).to(DEV),
                count=torch.from_numpy(cnt).to(DEV), count_np=cnt,
                X=torch.from_numpy(z['X'][idx]).to(DEV), desc=torch.from_numpy(z['desc'][idx]).to(DEV),
                tree=cKDTree(T[:, :2, 3]))
            print('map', cam, keep.sum(), 'frames', len(idx), 'points', f'risky {np.mean(risk(st[keep]) > max_gt_acc) * 100:.1f} %', flush=True)

    def select(self, cam, x, y, yaw, radius=8.0, n_pass=3, max_dyaw=0.6, pass_gap_ns=5e9):
        m = self.cams[cam]
        c = m['tree'].query_ball_point([x, y], radius)
        if not c:
            return np.zeros(0, int)
        c = np.array(c)
        c = c[np.abs(wrap(m['yaw'][c] - yaw)) < max_dyaw]
        if not len(c):
            return c
        c = c[np.argsort(m['stamp'][c])]
        g = np.concatenate([[0], np.cumsum(np.diff(m['stamp'][c]) > pass_gap_ns)])
        d = np.hypot(m['xy'][c, 0] - x, m['xy'][c, 1] - y)
        best = [c[g == k][np.argmin(d[g == k])] for k in range(g[-1] + 1)]
        # safe passes first (GT not near an accuracy spike), then by distance
        best = sorted(best, key=lambda i: (m['risk'][i] > self.max_gt_acc, np.hypot(m['xy'][i, 0] - x, m['xy'][i, 1] - y)))
        return np.array(best[:n_pass])

    def gather(self, cam, frames):
        m = self.cams[cam]
        fi = torch.from_numpy(frames).to(DEV)
        o, n = m['offset'][fi], m['count'][fi]
        tot = int(m['count_np'][frames].sum())
        start = torch.repeat_interleave(o - torch.cumsum(n, 0) + n, n, output_size=tot)   # output_size: no host sync
        idx = start + torch.arange(tot, device=DEV)
        return m['X'][idx], m['desc'][idx]


# ------------------------------------------------------------------ camera model (torch, batched over pose hypotheses)
class Cams:
    def __init__(self, cams, cal=None):
        cal = cal or data.calib()
        self.names = cams
        self.K = torch.tensor(np.stack([cal[c]['K'] for c in cams]), dtype=torch.float64, device=DEV)
        self.D = torch.tensor(np.stack([cal[c]['D'][:5] for c in cams]), dtype=torch.float64, device=DEV)
        Tbc = np.stack([cal[c]['T_base_cam'] for c in cams])
        Tcb = np.linalg.inv(Tbc)
        self.Rcb = torch.tensor(Tcb[:, :3, :3], device=DEV); self.tcb = torch.tensor(Tcb[:, :3, 3], device=DEV)
        self.W = cal[cams[0]]['width']; self.H = cal[cams[0]]['height']
        # max normalised radius that is inside the image (avoid the distortion polynomial folding back)
        self.rmax = []
        for c in cams:
            K, D = cal[c]['K'], cal[c]['D']
            corners = np.array([[0, 0], [self.W, 0], [0, self.H], [self.W, self.H]], np.float64)
            n = cv2.undistortPoints(corners.reshape(-1, 1, 2), K, D).reshape(-1, 2)
            self.rmax.append(float(np.linalg.norm(n, axis=1).max()) * 1.05)
        self.rmax = torch.tensor(self.rmax, dtype=torch.float64, device=DEV)
        self.f = 0.5 * (self.K[:, 0, 0] + self.K[:, 1, 1])
        self.K_np, self.D_np = self.K.cpu().numpy(), self.D.cpu().numpy()
        self.f_np = self.f.cpu().numpy()
        self.Rcb_np, self.tcb_np, self.rmax_np = Tcb[:, :3, :3], Tcb[:, :3, 3], self.rmax.cpu().numpy()

    def project(self, Xw, ci, Rwb, twb):
        """Xw (n,3) map points, ci (n,) camera index, Rwb (B,3,3), twb (B,3) base poses -> uv (B,n,2), ok (B,n)."""
        Xb = torch.einsum('bji,bnj->bni', Rwb, Xw[None] - twb[:, None])          # R^T (X - t)
        Xc = torch.einsum('nij,bnj->bni', self.Rcb[ci], Xb) + self.tcb[ci][None]
        z = Xc[..., 2]
        x, y = Xc[..., 0] / z.clamp(min=1e-3), Xc[..., 1] / z.clamp(min=1e-3)
        D = self.D[ci]
        k1, k2, p1, p2, k3 = D[:, 0], D[:, 1], D[:, 2], D[:, 3], D[:, 4]
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        K = self.K[ci]
        u = K[:, 0, 0] * xd + K[:, 0, 2]; v = K[:, 1, 1] * yd + K[:, 1, 2]
        ok = (z > 0.5) & (r2 < self.rmax[ci] ** 2)
        return torch.stack([u, v], -1), ok


def base_R(yaw, R_rp):
    c, s = torch.cos(yaw), torch.sin(yaw)
    Rz = torch.zeros(len(yaw), 3, 3, dtype=torch.float64, device=DEV)
    Rz[:, 0, 0] = c; Rz[:, 0, 1] = -s; Rz[:, 1, 0] = s; Rz[:, 1, 1] = c; Rz[:, 2, 2] = 1
    return Rz @ R_rp


def project_np(cams, Xw, ci, Rwb, twb):
    """numpy version of Cams.project for the solver: Rwb (B,3,3), twb (B,3) -> uv (B,n,2), ok (B,n)."""
    Xb = np.einsum('bji,bnj->bni', Rwb, Xw[None] - twb[:, None])
    Xc = np.einsum('nij,bnj->bni', cams.Rcb_np[ci], Xb) + cams.tcb_np[ci][None]
    z = Xc[..., 2]
    zc = np.maximum(z, 1e-3)
    x, y = Xc[..., 0] / zc, Xc[..., 1] / zc
    D = cams.D_np[ci]
    k1, k2, p1, p2, k3 = D[:, 0], D[:, 1], D[:, 2], D[:, 3], D[:, 4]
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    K = cams.K_np[ci]
    uv = np.stack([K[:, 0, 0] * xd + K[:, 0, 2], K[:, 1, 1] * yd + K[:, 1, 2]], -1)
    return uv, (z > 0.5) & (r2 < cams.rmax_np[ci] ** 2)


def pose_R(p, R_rp):
    """Base rotation for p = (x, y, yaw[, droll, dpitch, dz]): Rz(yaw) R_rp Rx(droll) Ry(dpitch)."""
    c, s = np.cos(p[2]), np.sin(p[2])
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ R_rp
    if len(p) > 3:
        cr, sr, cp, sp = np.cos(p[3]), np.sin(p[3]), np.cos(p[4]), np.sin(p[4])
        R = R @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return R


def pnp_lin(cams, Xw, ci, uv, p, z, R_rp, jac=True):
    """Residuals (projection - observation, px) and analytic Jacobian (n,2,nd) for the pose parameters p."""
    nd = len(p)
    c, s = np.cos(p[2]), np.sin(p[2])
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    Rx = Ry = np.eye(3)
    if nd > 3:
        cr, sr, cp, sp = np.cos(p[3]), np.sin(p[3]), np.cos(p[4]), np.sin(p[4])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]); Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    t = np.array([p[0], p[1], z + (p[5] if nd > 3 else 0.0)])
    Y = (Xw - t) @ Rz                      # Rz^T v (row vectors)
    W = Y @ R_rp                           # R_rp^T Y
    V = W @ Rx
    Xb = V @ Ry
    Rcb, tcb = cams.Rcb_np[ci], cams.tcb_np[ci]
    Xc = np.einsum('nij,nj->ni', Rcb, Xb) + tcb
    zc = np.maximum(Xc[:, 2], 1e-3)
    x, y = Xc[:, 0] / zc, Xc[:, 1] / zc
    D = cams.D_np[ci]
    k1, k2, p1, p2, k3 = D[:, 0], D[:, 1], D[:, 2], D[:, 3], D[:, 4]
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    K = cams.K_np[ci]
    fx, fy = K[:, 0, 0], K[:, 1, 1]
    r = np.stack([fx * xd + K[:, 0, 2], fy * yd + K[:, 1, 2]], 1) - uv
    ok = (Xc[:, 2] > 0.5) & (r2 < cams.rmax_np[ci] ** 2)
    if not jac:
        return r, ok, None
    Rwb_T = (Rz @ R_rp @ Rx @ Ry).T                                   # d Xb / d t = -Rwb^T
    A = (R_rp @ Rx @ Ry).T                                            # Xb = A Y
    dXb = np.zeros((len(Xw), 3, nd))
    dXb[:, :, 0] = -Rwb_T[:, 0]; dXb[:, :, 1] = -Rwb_T[:, 1]
    dY = np.stack([Y[:, 1], -Y[:, 0], np.zeros(len(Y))], 1)          # -e_z x Y
    dXb[:, :, 2] = dY @ A.T
    if nd > 3:
        dV = np.stack([np.zeros(len(V)), V[:, 2], -V[:, 1]], 1)       # -e_x x V
        dXb[:, :, 3] = dV @ Ry
        dXb[:, :, 4] = np.stack([-Xb[:, 2], np.zeros(len(Xb)), Xb[:, 0]], 1)   # -e_y x Xb
        dXb[:, :, 5] = -Rwb_T[:, 2]
    dXc = np.matmul(Rcb, dXb)                                         # (n,3,nd)
    iz = 1.0 / zc
    dx = (dXc[:, 0] - x[:, None] * dXc[:, 2]) * iz[:, None]           # (n,nd)
    dy = (dXc[:, 1] - y[:, None] * dXc[:, 2]) * iz[:, None]
    drad = k1 + 2 * k2 * r2 + 3 * k3 * r2 ** 2
    a11 = rad + 2 * x * x * drad + 2 * p1 * y + 6 * p2 * x
    a12 = 2 * x * y * drad + 2 * p1 * x + 2 * p2 * y
    a21 = 2 * x * y * drad + 2 * p1 * x + 2 * p2 * y
    a22 = rad + 2 * y * y * drad + 6 * p1 * y + 2 * p2 * x
    J = np.stack([fx[:, None] * (a11[:, None] * dx + a12[:, None] * dy),
                  fy[:, None] * (a21[:, None] * dx + a22[:, None] * dy)], 1)
    return r, ok, J


def solve_pnp(cams, Xw, ci, uv, p0, z, R_rp, huber=2.0, iters=8, inl_px=3.0, extra_dof=0, prior=(0.01, 0.01, 0.1)):
    """Robust Gauss-Newton on (x, y, yaw) over all cameras (numpy, analytic Jacobian).
    extra_dof = 3 also estimates (droll, dpitch, dz) w.r.t. the map frame's roll/pitch/height, with Gaussian priors
    `prior` (rad, rad, m) weighted against 1 px of reprojection error. Huber IRLS, then a hard inlier set (< inl_px)
    for the second half. Returns full parameters, covariance of (x, y, yaw) (marginal, inliers), inlier mask,
    residual sigma [px]."""
    nd = 3 + extra_dof
    p = np.zeros(nd); p[:min(nd, len(p0))] = np.asarray(p0)[:nd]
    lam = np.zeros(nd)
    if extra_dof:
        lam[3:] = 1.0 / np.asarray(prior[:extra_dof]) ** 2
    w_in = np.ones(len(Xw))
    for it in range(iters):
        r0, ok, J = pnp_lin(cams, Xw, ci, uv, p, z, R_rp)
        e = np.sqrt((r0 ** 2).sum(1))
        if it == iters // 2:
            w_in = (e < inl_px).astype(np.float64)
        w = np.where(e < huber, 1.0, huber / np.maximum(e, 1e-9)) * ok * w_in
        Jw = (J * np.sqrt(w)[:, None, None]).reshape(-1, nd)
        A = Jw.T @ Jw + np.diag(lam)
        bb = Jw.T @ (r0 * np.sqrt(w)[:, None]).reshape(-1) + lam * p
        try:
            dp = -np.linalg.solve(A + 1e-9 * np.eye(nd), bb)
        except np.linalg.LinAlgError:
            break
        p = p + dp
        if np.hypot(dp[0], dp[1]) < 1e-4 and abs(dp[2]) < 1e-6 and it >= iters // 2:
            break
    r0, ok, J = pnp_lin(cams, Xw, ci, uv, p, z, R_rp)
    e = np.sqrt((r0 ** 2).sum(1))
    inl = (e < inl_px) & ok
    n = int(inl.sum())
    if n < 6:
        return p, np.eye(3) * 1e6, inl, float('nan')
    s2 = float((e[inl] ** 2).sum() / max(2 * n - nd, 1))
    Ji = J[inl].reshape(-1, nd)
    cov = np.linalg.inv(Ji.T @ Ji + np.diag(lam) * s2)[:3, :3] * s2
    return p, cov, inl, float(np.sqrt(s2))


def guided_match_mask(uv_m, ok_m, dm, kq, dq, radius, sim_thr):
    """Mutual-best descriptor match within `radius` px of each projected map point; no host sync.
    Returns (good mask over map points, index of the matched query keypoint)."""
    if len(uv_m) == 0 or len(kq) == 0:
        return torch.zeros(len(uv_m), dtype=torch.bool, device=DEV), torch.zeros(len(uv_m), dtype=torch.long, device=DEV)
    d = torch.cdist(uv_m.float(), kq.float())
    S = dm.float() @ dq.float().T
    S = torch.where((d < radius) & ok_m[:, None], S, torch.full_like(S, -2))
    best, j = S.max(1)
    _, i = S.max(0)
    good = (best > sim_thr) & (i[j] == torch.arange(len(S), device=DEV))
    return good, j


def guided_match(uv_m, ok_m, dm, kq, dq, radius, sim_thr):
    """uv_m (M,2) projected map points, dm (M,256), kq (Kq,2) query keypoints, dq (Kq,256). Mutual best within radius."""
    if len(uv_m) == 0 or len(kq) == 0:
        return torch.zeros(0, dtype=torch.long, device=DEV), torch.zeros(0, dtype=torch.long, device=DEV)
    d = torch.cdist(uv_m.float(), kq.float())
    S = dm.float() @ dq.float().T
    S = torch.where((d < radius[:, None] if torch.is_tensor(radius) else d < radius) & ok_m[:, None], S, torch.full_like(S, -2))
    best, j = S.max(1)
    _, i = S.max(0)
    good = (best > sim_thr) & (i[j] == torch.arange(len(S), device=DEV))
    im = torch.nonzero(good).squeeze(1)
    return im, j[im]


# ------------------------------------------------------------------ query side
def augment(im, aug):
    """Photometric stress test on query images (robustness to the test run's light / speed), e.g. 'gamma2',
    'gain0.4', 'blur15' (horizontal motion blur, px), 'noise8' (Gaussian sigma), combinable with '+'."""
    rng = np.random.default_rng(int(im[::97, ::89].sum()))
    for a in aug.split('+'):
        if a.startswith('gamma'):
            im = (255 * (im / 255.0) ** float(a[5:])).astype(np.uint8)
        elif a.startswith('gain'):
            im = np.clip(im.astype(np.float32) * float(a[4:]), 0, 255).astype(np.uint8)
        elif a.startswith('blur'):
            k = int(a[4:]); im = cv2.filter2D(im, -1, np.ones((1, k), np.float32) / k)
        elif a.startswith('noise'):
            im = np.clip(im + rng.normal(0, float(a[5:]), im.shape), 0, 255).astype(np.uint8)
    return im


def sp_batch(sp, image):
    """SuperPoint (LightGlue implementation) on a batch; returns per-image lists (the original stacks them, which
    fails when the images have different keypoint counts). Descriptors are (K, 256)."""
    from lightglue.superpoint import sample_descriptors, simple_nms, top_k_keypoints
    r = sp.relu
    x = r(sp.conv1b(r(sp.conv1a(image)))); x = sp.pool(x)
    x = r(sp.conv2b(r(sp.conv2a(x)))); x = sp.pool(x)
    x = r(sp.conv3b(r(sp.conv3a(x)))); x = sp.pool(x)
    x = r(sp.conv4b(r(sp.conv4a(x))))
    scores = torch.nn.functional.softmax(sp.convPb(r(sp.convPa(x))).float(), 1)[:, :-1]
    b, _, h, w = scores.shape
    scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8).permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
    scores = simple_nms(scores, sp.conf.nms_radius)
    pad = sp.conf.remove_borders
    if pad:
        scores[:, :pad] = -1; scores[:, :, :pad] = -1; scores[:, -pad:] = -1; scores[:, :, -pad:] = -1
    desc = torch.nn.functional.normalize(sp.convDb(r(sp.convDa(x))).float(), p=2, dim=1)
    kps, des = [], []
    for i in range(b):
        k = torch.nonzero(scores[i] > sp.conf.detection_threshold)
        s = scores[i][k[:, 0], k[:, 1]]
        k, s = top_k_keypoints(k, s, sp.conf.max_num_keypoints)
        k = torch.flip(k, [1]).float()
        kps.append(k)
        des.append(sample_descriptors(k[None], desc[i:i + 1], 8)[0].T)
    return kps, des


class Frontend:
    def __init__(self, cams, max_kp=2048, threads=7, resize=1024, cal=None):
        from lightglue import SuperPoint
        self.cams = cams
        self.resize = resize
        self.aug = None
        self.mask_bright = 0
        self.kp_keep, self.n_calls = 1.0, 0
        self.gpu_decode = True
        self.sp = SuperPoint(max_num_keypoints=max_kp).eval().to(DEV)
        self.pool = ThreadPoolExecutor(threads)
        cal = cal or data.calib()
        self.ycut = [int(CROP_BOTTOM[c] * cal[c]['height']) for c in cams]

    def load(self, paths):
        """JPEG files -> grayscale images: nvJPEG batch decode on the GPU (default) or OpenCV on CPU threads."""
        paths = [p if isinstance(p, str) else None for p in paths]       # missing frame -> None (pandas gives NaN)
        if self.gpu_decode and not self.aug:
            from torchvision.io import ImageReadMode, decode_jpeg
            raw = list(self.pool.map(lambda p: torch.frombuffer(bytearray(open(p, 'rb').read()), dtype=torch.uint8) if p else None, paths))
            ok = [i for i, r in enumerate(raw) if r is not None]
            dec = decode_jpeg([raw[i] for i in ok], mode=ImageReadMode.GRAY, device=DEV)
            ims = [None] * len(paths)
            for i, d in zip(ok, dec):
                ims[i] = d[0]
            return ims
        ims = list(self.pool.map(lambda p: cv2.imread(p, cv2.IMREAD_GRAYSCALE) if p else None, paths))
        if self.aug:
            ims = [augment(im, self.aug) if im is not None else None for im in ims]
        return ims

    @torch.no_grad()
    def features(self, ims):
        ok = [i for i, im in enumerate(ims) if im is not None]
        if torch.is_tensor(ims[ok[0]]):
            x = torch.stack([ims[i] for i in ok])
        else:
            x = torch.from_numpy(np.stack([ims[i] for i in ok])).to(DEV)
        x = x.float().div_(255)[:, None]
        for b, i in enumerate(ok):
            x[b, :, self.ycut[i]:] = 0                                    # car body -> no keypoints
        if self.resize:                                                   # same preprocessing as the map (extract())
            from lightglue.utils import ImagePreprocessor
            x, scale = ImagePreprocessor(resize=self.resize)(x)
        with torch.autocast('cuda', dtype=torch.float16):
            kps, des = sp_batch(self.sp, x)
        if self.mask_bright:          # no keypoints on or next to light sources (lamps: night only, often repetitive)
            bm = torch.nn.functional.max_pool2d((x >= self.mask_bright / 255.0).float(), 9, 1, 4)[:, 0] > 0
        out = [None] * len(ims)
        for b, i in enumerate(ok):
            kp, de = kps[b], des[b]
            keep_b = ~bm[b][kp[:, 1].long().clamp(0, bm.shape[1] - 1), kp[:, 0].long().clamp(0, bm.shape[2] - 1)] \
                if self.mask_bright else None
            if self.resize:
                kp = (kp + 0.5) / scale[None] - 0.5
            keep = kp[:, 1] < self.ycut[i] - 6
            if keep_b is not None:
                keep &= keep_b
            if self.kp_keep < 1:      # stress test: keep a random share of the keypoints (fewer correct matches)
                self.n_calls += 1
                g = torch.Generator(device=DEV).manual_seed(self.n_calls)
                keep &= torch.rand(len(kp), generator=g, device=DEV) < self.kp_keep
            out[i] = (kp[keep].double(), de[keep].half())
        return out


# ------------------------------------------------------------------ car-state dead reckoning
class CarState:
    def __init__(self, s=None):
        s = data.allowed_state() if s is None else s
        self.t = s.timestamp_ns.values
        self.vx, self.vy, self.wz, self.v = s.vx_mps.values, s.vy_mps.values, s.wz_radps.values, s.v_mps.values

    def integrate(self, ta, tb, t_avail):
        """Motion of the base from time ta to tb (ns) in the frame at ta, using only samples <= t_avail.
        Piecewise-constant (zero-order hold of the latest sample) integration of vx, vy, wz."""
        i0 = max(np.searchsorted(self.t, ta, 'right') - 1, 0)
        i1 = np.searchsorted(self.t, min(tb, t_avail), 'right')
        knots = np.concatenate([[ta], self.t[i0 + 1:i1], [tb]]).astype(np.float64)
        knots = knots[(knots >= ta) & (knots <= tb)]
        idx = np.clip(np.searchsorted(self.t, knots[:-1], 'right') - 1, 0, len(self.t) - 1)
        idx = np.minimum(idx, max(i1 - 1, 0))                             # never a sample after t_avail
        dt = np.diff(knots) / 1e9
        th = np.concatenate([[0], np.cumsum(self.wz[idx] * dt)])
        thm = 0.5 * (th[:-1] + th[1:])
        dx = np.sum((np.cos(thm) * self.vx[idx] - np.sin(thm) * self.vy[idx]) * dt)
        dy = np.sum((np.sin(thm) * self.vx[idx] + np.cos(thm) * self.vy[idx]) * dt)
        return dx, dy, th[-1]

    def at(self, t):
        i = max(np.searchsorted(self.t, t, 'right') - 1, 0)
        return self.v[i], self.wz[i]


class Reloc:
    """Global relocalisation for the lost case: MegaLoc descriptors of the query image set (averaged cosine similarity
    over cameras) against the map run (training images, descriptors precomputed by odo.vpr), then up to `n_cand`
    spatially distinct candidates are verified with wide-radius guided matching + PnP (measure())."""

    def __init__(self, cams, windows, desc_dir='outputs/10_vpr_desc', dt_ns=0):
        from .vpr import load_model
        from .retrieval import aligned
        self.model = load_model('megaloc')
        self.cams = cams
        T, D = [], []
        import os as _os
        for split in ('train', 'val', 'L8'):           # descriptor files cover 0-897, 897-1120 and L8
            if not _os.path.exists(f'{desc_dir}/megaloc_fl_{split}.npz'):
                continue
            t, d = aligned(desc_dir, 'megaloc', cams, split, ref='fl')
            T.append(t); D.append(np.stack(d, 0))
        t = np.concatenate(T); Dd = np.concatenate(D, 1)
        keep = np.any([(t >= a) & (t < b) for a, b in windows], axis=0)
        self.t = t[keep]
        self.D = torch.from_numpy(Dd[:, keep]).to(DEV)                    # (cams, N, d) fp16
        x, y, yaw = data.interp_gt(self.t + dt_ns)
        self.pose = np.stack([x, y, yaw], 1)
        print('reloc db', len(self.t), 'frames', flush=True)

    @torch.no_grad()
    def candidates(self, paths, n_cand=4, min_sep=8.0, prior=None, near_m=10.0):
        """Spatially distinct top candidates (poses). Also sets self.last (report): MegaLoc similarity of the top-1
        database frame and its position, and the best similarity among database frames within near_m of `prior`."""
        from .vpr import MEAN, STD, load_img
        q = []
        for c, p in zip(self.cams, paths):
            if not isinstance(p, str):
                q.append(None); continue
            q.append(load_img(p, c, 322))
        ok = [i for i, im in enumerate(q) if im is not None]
        S = 0
        for shape in {q[i].shape for i in ok}:                         # crops differ per camera: batch by shape
            grp = [i for i in ok if q[i].shape == shape]
            x = torch.from_numpy(np.stack([q[i] for i in grp])).permute(0, 3, 1, 2).float().div(255)
            x = ((x - MEAN) / STD).to(DEV)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                d = torch.nn.functional.normalize(self.model(x).float(), dim=-1)
            S = S + sum(self.D[i].float() @ d[k] for k, i in enumerate(grp))
        S = S / len(ok)
        top = torch.topk(S, 50).indices.cpu().numpy()
        self.last = dict(ml_sim_top1=float(S[top[0]]), ml_top1_x=float(self.pose[top[0], 0]),
                         ml_top1_y=float(self.pose[top[0], 1]))
        if prior is not None:
            near = np.hypot(self.pose[:, 0] - prior[0], self.pose[:, 1] - prior[1]) < near_m
            self.last['ml_sim_prior'] = float(S[torch.from_numpy(near).to(S.device)].max()) if near.any() else np.nan
        cands = []
        for j in top:
            if all(np.hypot(*(self.pose[j, :2] - self.pose[c, :2])) > min_sep for c in cands):
                cands.append(j)
            if len(cands) == n_cand:
                break
        return [self.pose[j] for j in cands]


# ------------------------------------------------------------------ main loop
def measure(fmap, cm, feats, X, cfg, r1, frames=None):
    """Map-based fix around the prior pose X (two guided-matching + PnP stages, the second with a small radius around
    the first solution). frames: optional {cam: map frame indices} override. Returns (x,y,yaw or None, cov, info)."""
    cams = cm.names
    info = dict(status='none', n_m1=0, n_inl=0)
    Xs, Ds, sl, nearest, o = [], [], [], None, 0
    for i, cam in enumerate(cams):
        if feats[i] is None:
            continue
        fr = frames[cam] if frames is not None else fmap.select(cam, X[0], X[1], X[2], cfg['sel_radius'], cfg['n_pass'])
        if not len(fr):
            continue
        if nearest is None:
            nearest = fmap.cams[cam]['T'][fr[0]]
            xy = fmap.cams[cam]['xy'][fr]                                             # off the mapped lines?
            info['map_dist'] = float(np.hypot(xy[:, 0] - X[0], xy[:, 1] - X[1]).min())
        Xm, Dm = fmap.gather(cam, fr)
        n = int(fmap.cams[cam]['count_np'][fr].sum())
        Xs.append(Xm); Ds.append(Dm); sl.append((i, o, o + n)); o += n
    if not Xs:
        info['nomap'] = True              # no map frame near the prediction: off the mapped lines, not "lost"
        return None, None, info
    Xm = torch.cat(Xs).double(); Dm = torch.cat(Ds)
    ci_np = np.concatenate([np.full(b - a, i) for i, a, b in sl])
    ci = torch.from_numpy(ci_np).to(DEV)
    z = nearest[2, 3]
    yawA = np.arctan2(nearest[1, 0], nearest[0, 0])
    R_rp = Rotation.from_euler('z', -yawA).as_matrix() @ nearest[:3, :3]
    nd = 3 + cfg['extra_dof']
    p = np.zeros(nd); p[:3] = X
    prior = (np.radians(cfg['prior_rp_deg']),) * 2 + (cfg['prior_z'],)
    for stage, (rad, sim) in enumerate([(r1, cfg['sim1']), (cfg['r2'], cfg['sim2'])]):
        R = torch.tensor(pose_R(p, R_rp), device=DEV)[None]
        t = torch.tensor([[p[0], p[1], z + (p[5] if nd > 3 else 0.0)]], dtype=torch.float64, device=DEV)
        uv, okp = cm.project(Xm, ci, R, t)
        uv, okp = uv[0], okp[0]
        okp &= (uv[:, 0] >= 0) & (uv[:, 0] < cm.W) & (uv[:, 1] >= 0) & (uv[:, 1] < cm.H)
        goods, js, kqs = [], [], []
        for i, a, b in sl:
            kq, dq = feats[i]
            good, j = guided_match_mask(uv[a:b], okp[a:b], Dm[a:b], kq, dq, rad * float(cm.f_np[i]) / 1000.0, sim)
            goods.append(good); js.append(j); kqs.append(kq)
        good = torch.cat(goods)
        uq_all = torch.cat([kq[j] for kq, j in zip(kqs, js)])
        mi = torch.nonzero(good).squeeze(1)
        mi_np = mi.cpu().numpy()
        if stage == 0 and cfg['sim1_lo'] and len(mi_np) < cfg['lo_below']:
            # few descriptor matches (e.g. appearance change night -> day): accept weaker similarities, the
            # projection window and the robust PnP do the disambiguation
            goods, js = [], []
            for i, a, b in sl:
                kq, dq = feats[i]
                good, j = guided_match_mask(uv[a:b], okp[a:b], Dm[a:b], kq, dq, rad * float(cm.f_np[i]) / 1000.0, cfg['sim1_lo'])
                goods.append(good); js.append(j)
            good = torch.cat(goods)
            uq_all = torch.cat([kq[j] for kq, j in zip(kqs, js)])
            mi = torch.nonzero(good).squeeze(1)
            mi_np = mi.cpu().numpy()
            info['lo'] = True
        if stage == 0:
            info['n_m1'] = len(mi_np)
            info['m1_cam'] = '/'.join(str(int(c)) for c in np.bincount(ci_np[mi_np], minlength=len(cams)))
        if len(mi_np) < cfg['min_inl']:
            break
        cin = ci_np[mi_np]
        p, cov, inl, rms = solve_pnp(cm, Xm[mi].cpu().numpy(), cin, uq_all[mi].cpu().numpy(), p, z, R_rp,
                                     huber=cfg['huber'], extra_dof=cfg['extra_dof'], prior=prior)
        info.update(n_inl=int(inl.sum()), rms_px=rms, stage=stage)
        if stage == 1:
            info['inl_cam'] = '/'.join(str(int(inl[cin == i].sum())) for i in range(len(cams)))
            if nd > 3:
                info.update(droll=float(np.degrees(p[3])), dpitch=float(np.degrees(p[4])), dz=float(p[5]))
            if int(inl.sum()) >= cfg['min_inl']:
                return p[:3].copy(), cov, info
    return None, None, info


def move(X, dx, dy, dth):
    c, s = np.cos(X[2]), np.sin(X[2])
    return np.array([X[0] + c * dx - s * dy, X[1] + s * dx + c * dy, wrap(X[2] + dth)])


DEFAULT = dict(n_pass=3, sel_radius=15.0, r1=24.0, r2=4.0, sim1=0.75, sim2=0.6, min_inl=30, huber=2.0,
               q_lon=0.01, q_lat=0.005, q_yaw=0.0005, q0=0.005, r_scale=4.0, r0_pos=0.10, r0_yaw=0.004,
               gate=16.0, chal_n=8, chal_m=0.3, chal_yaw=0.005, max_kp=2048, r1_grow=40.0, max_gt_acc=1.0,
               extra_dof=3, prior_rp_deg=0.5, prior_z=0.05, reloc=True, reloc_after=1.0, reloc_every=0.5,
               reloc_r1=60.0, reloc_min_inl=80, reloc_sig0=0.5, reloc_sig_rate=0.3, reloc_strong_inl=300,
               r_inl_ref=0, chal_fixed=True, r_dist_ref=0, sim1_lo=0.0, lo_below=300, mask_bright=0, kp_keep=1.0,
               reloc_yaw0_deg=0.3, reloc_yaw_rate_deg=0.1, yaw_min_inl=60,   # Exp24 (day transfer); Exp21: 1.0, 2.0, 0 and reloc_sig_rate 2.0
               log_sim=False)   # report only: MegaLoc similarity on every frame (outside the timed loop, no effect on poses)


def image_sets(split, cams, qrun=None):
    """One row per trigger (fl timeline): image path per camera (None if that camera has no frame within 5 ms)."""
    ref = data.frames_split('fl', split) if qrun is None else qrun.frames('fl')
    out = {'stamp_ns': ref.stamp_ns.values}
    for c in cams:
        f = data.frames(c) if qrun is None else qrun.frames(c); st = f.stamp_ns.values
        j = np.clip(np.searchsorted(st, ref.stamp_ns.values), 1, len(st) - 1)
        j = np.where(np.abs(st[j - 1] - ref.stamp_ns.values) < np.abs(st[j] - ref.stamp_ns.values), j - 1, j)
        ok = np.abs(st[j] - ref.stamp_ns.values) < 5e6
        out[c] = np.where(ok, f.path.values[j], None)
    return pd.DataFrame(out)


def run(split, out, map_dir='outputs/20_map', cams=tuple(data.CAMS), limit=None, cfg=None, verbose=True, query_root=None,
        map_until_s=None, fold=None):
    """split: 'val' (map = t < 897 s), 'tune'/'dev2' (map = train minus the lap +-10 s) or 'test' (query_root =
    an extracted run, e.g. the hidden test run; map = the whole training run; init = its first pose only)."""
    cfg = dict(DEFAULT, **(cfg or {}))
    cams = list(cams)
    os.makedirs(out, exist_ok=True)
    qrun = data.Run(query_root) if query_root else None
    t0 = data.t0_ns(); a, b = data.SPLITS.get(split, (0, 0))
    if fold is not None:              # cross-validation (docs/laps.md): split = a lap, map = the fold's map laps
        assert split not in data.FOLDS[fold]['map_laps'] or (data.FOLDS[fold].get('day_query') and cfg.get('img_root')), \
            f'{split} is in the map of {fold}'
        windows = data.fold_windows(fold)
    elif qrun is not None:
        windows = [(0, 2 ** 62) if map_until_s is None else (0, t0 + int(map_until_s * 1e9))]
    elif split == 'val':
        windows = [(0, t0 + int(897e9))]
    elif split == 'test':
        windows = [(0, 2 ** 62)]
    else:  # tune lap: map = train without the lap +- 10 s
        windows = [(0, t0 + int((a - 10) * 1e9)), (t0 + int((b + 10) * 1e9), t0 + int(897e9))]
    fmap = FeatureMap(map_dir, cams, windows, cfg['max_gt_acc'])
    qcal = qrun.calib() if qrun is not None else None
    torch.set_num_threads(4); cv2.setNumThreads(2)
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)                      # tiny BLAS calls: threading only adds overhead
    cm = Cams(cams, qcal)
    fe = Frontend(cams, cfg['max_kp'], cal=qcal)
    fe.aug = cfg.get('aug')
    fe.mask_bright = cfg['mask_bright']
    fe.kp_keep = cfg['kp_keep']
    cs = CarState(qrun.allowed_state() if qrun is not None else None)
    sets = image_sets(split, cams, qrun)
    if limit:
        sets = sets.iloc[:limit]
    if cfg.get('img_root'):                   # same frames, other pixels (e.g. the day proxy, scripts/day_translate.py)
        for c in cams:
            sets[c] = [p.replace(data.DATA, cfg['img_root']) if isinstance(p, str) else None for p in sets[c]]
    dt_ns = int(np.load(f'{map_dir}/{cams[0]}.npz')['dt_ns'])
    reloc = Reloc(cams, windows, dt_ns=dt_ns) if cfg['reloc'] else None
    last_reloc = -1e18

    # init: first pose of the run / segment (allowed), expressed at GT-clock time t + dt
    st0 = int(sets.stamp_ns[0])
    if cfg.get('init_pose'):                  # (stamp_ns or None, x, y, yaw): --first_pose FILE / --no_first_pose
        tp, x0, y0, yaw0 = cfg['init_pose']
    elif qrun is not None:
        tp, x0, y0, yaw0 = qrun.first_pose()
    else:
        tp, (x0, y0, yaw0) = st0 + dt_ns, data.first_pose(st0 + dt_ns)
    tp = st0 + dt_ns if tp is None else int(tp)   # no stamp: the pose at the first image
    if st0 + dt_ns > tp:                      # propagated to the first image by dead reckoning
        x0, y0, yaw0 = move(np.array([x0, y0, yaw0]), *cs.integrate(tp, st0 + dt_ns, st0))
    s0, s0yaw = cfg.get('init_sigma_xy', 0.05), np.radians(cfg.get('init_sigma_yaw_deg', np.degrees(0.002)))
    X = np.array([x0, y0, yaw0]); P = np.diag([s0 ** 2, s0 ** 2, s0yaw ** 2])
    kid = cfg.get('kidnap')                   # robustness test: (t since start [s], lon [m], lat [m], yaw [deg])
    if cfg.get('init_offset'):
        kid = [0.0] + list(cfg['init_offset'])
    t_state = st0 + dt_ns
    rows, infos = [], []
    n_rej = 0; last_fix = st0
    chal, n_chal = None, 0       # challenger hypothesis seeded by gated fixes
    times = []
    # warm-up (cuDNN autotune, allocator)
    fe.features(fe.load(list(sets.iloc[0][cams])))
    for k in range(len(sets)):
        torch.cuda.synchronize(); tic = time.perf_counter()
        stamp = int(sets.stamp_ns[k]); tm = stamp + dt_ns
        # 1. predict
        if tm > t_state:
            dx, dy, dth = cs.integrate(t_state, tm, stamp)
            c, s = np.cos(X[2]), np.sin(X[2])
            ds = np.hypot(dx, dy)
            X = move(X, dx, dy, dth)
            if chal is not None:
                chal = move(chal, dx, dy, dth)
            F = np.array([[1, 0, -(s * dx + c * dy)], [0, 1, c * dx - s * dy], [0, 0, 1]])
            Rw = np.array([[np.cos(X[2]), -np.sin(X[2])], [np.sin(X[2]), np.cos(X[2])]])
            Ql = np.diag([(cfg['q_lon'] * ds + cfg['q0']) ** 2, (cfg['q_lat'] * ds + cfg['q0']) ** 2])
            Q = np.zeros((3, 3)); Q[:2, :2] = Rw @ Ql @ Rw.T
            Q[2, 2] = (cfg['q_yaw'] + 0.01 * abs(dth)) ** 2
            P = F @ P @ F.T + Q
            t_state = tm
        if kid is not None and (stamp - st0) / 1e9 >= kid[0]:
            c_, s_ = np.cos(X[2]), np.sin(X[2])
            X = X + np.array([c_ * kid[1] - s_ * kid[2], s_ * kid[1] + c_ * kid[2], np.radians(kid[3])])
            kid = None                        # the filter keeps its (now wrong) confidence
        # 2-4. measurement
        ims = fe.load(list(sets.iloc[k][cams]))
        feats = fe.features(ims)
        since = (stamp - last_fix) / 1e9
        r1 = cfg['r1'] + cfg['r1_grow'] * min(since, 2.0)
        p_meas, cov_meas, info = measure(fmap, cm, feats, X, cfg, r1)
        relocated = False
        if reloc is not None and since > cfg['reloc_after'] and (stamp - last_reloc) / 1e9 >= cfg['reloc_every'] and \
                not info.get('nomap') and (p_meas is None or info.get('n_inl', 0) < cfg['reloc_min_inl'] or n_rej > 0):
            # lost: verify retrieval candidates (and the current estimate) with a wide search radius
            last_reloc = stamp
            best = None
            cands = reloc.candidates(list(sets.iloc[k][cams]), prior=X)
            ml_info = dict(reloc.last)
            for prior in [X] + cands:
                pr, cr, ir = measure(fmap, cm, feats, np.asarray(prior, dtype=np.float64), cfg, cfg['reloc_r1'])
                if pr is not None and (best is None or ir['n_inl'] > best[2]['n_inl']):
                    best = (pr, cr, ir)
            if best is not None and best[2]['n_inl'] >= cfg['reloc_min_inl']:
                p_meas, cov_meas, info = best
                relocated = True
            info['reloc_try'] = 1                     # global retrieval attempted at this frame (report: rates)
            info['reloc_best_inl'] = best[2]['n_inl'] if best is not None else 0
            info.update(ml_info)
        # 5. update
        if p_meas is not None:
            Rm = cov_meas * cfg['r_scale'] + np.diag([cfg['r0_pos'] ** 2, cfg['r0_pos'] ** 2, cfg['r0_yaw'] ** 2])
            if cfg['r_inl_ref']:          # weak fixes (few inliers) count less
                Rm = Rm * max(1.0, cfg['r_inl_ref'] / max(info['n_inl'], 1))
            if cfg['r_dist_ref']:         # fixes far from the mapped lines count less (trust odometry more there)
                Rm = Rm * (1.0 + (info.get('map_dist', 0.0) / cfg['r_dist_ref']) ** 2)
            Rm_reset = Rm                 # covariance after a jump (reloc / challenger reset) keeps the fix's yaw
            if info['n_inl'] < cfg['yaw_min_inl']:
                # weak fix: position only. A PnP yaw from few inliers is less reliable than the car-state yaw rate
                # (Exp24: a 30-inlier fix moved yaw by 1.6 deg, the filter then drifted 11 m)
                Rm = Rm.copy(); Rm[2, :2] = Rm[:2, 2] = 0.0; Rm[2, 2] = np.radians(30.0) ** 2
            nu = p_meas - X; nu[2] = wrap(nu[2])
            S = P + Rm
            d2 = float(nu @ np.linalg.solve(S, nu))
            info.update(d2=d2, mx=p_meas[0], my=p_meas[1], myaw=p_meas[2])
            # innovation = fix minus the dead-reckoning prediction, in the predicted car frame (report: run_report)
            cy, sy = np.cos(X[2]), np.sin(X[2])
            info.update(innov=float(np.hypot(nu[0], nu[1])), innov_lon=float(cy * nu[0] + sy * nu[1]),
                        innov_lat=float(-sy * nu[0] + cy * nu[1]), innov_yaw_deg=float(np.degrees(nu[2])))
            # a relocalisation jump must be plausible for the time without a fix (dead reckoning drifts slowly),
            # unless the evidence is very strong. Dead reckoning drifts ~0.06 m/s (~10 m per lap); 0.3 m/s keeps a 5x margin
            # (Exp24: with 2 m/s a day-image run accepted a wrong 21 m jump after 5.7 s without a fix)
            sig = cfg['reloc_sig0'] + cfg['reloc_sig_rate'] * since
            # yaw: car-state dead reckoning drifts ~0.09 deg in 1 s, ~0.4 deg in 10 s (stage 1), so a large yaw jump is
            # implausible however long the filter went without a fix (Exp24: a 2.5 deg wrong jump caused 7 m drift)
            sig_yaw = np.radians(cfg['reloc_yaw0_deg'] + cfg['reloc_yaw_rate_deg'] * since)
            d2_loose = float(nu @ np.linalg.solve(S + np.diag([sig ** 2, sig ** 2, sig_yaw ** 2]), nu))
            if relocated and d2 >= cfg['gate'] and d2_loose >= cfg['gate'] and info['n_inl'] < cfg['reloc_strong_inl']:
                info['status'] = 'reloc_rejected'; relocated = False; n_rej += 1
            elif relocated and d2 >= cfg['gate']:
                X = p_meas.copy(); P = Rm_reset + np.diag([0.1 ** 2, 0.1 ** 2, 0.003 ** 2])
                info['status'] = 'reloc'; n_rej = 0; last_fix = stamp; chal, n_chal = None, 0
            elif d2 < cfg['gate'] and info['status'] != 'reloc_rejected':
                Kg = P @ np.linalg.inv(S)
                X = X + Kg @ nu; X[2] = wrap(X[2]); P = (np.eye(3) - Kg) @ P
                info['status'] = 'fix'; n_rej = 0; last_fix = stamp
                chal, n_chal = None, 0
            else:
                n_rej += 1; info['status'] = 'gated'
                # challenger: gated fixes that agree with each other (through dead reckoning) take over
                # agreement is checked against the challenger's first fix propagated by dead reckoning (chal_fixed),
                # so a slowly drifting sequence of fixes does not count as consistent
                if chal is not None and np.hypot(*(p_meas[:2] - chal[:2])) < cfg['chal_m'] and \
                        abs(wrap(p_meas[2] - chal[2])) < cfg['chal_yaw']:
                    n_chal += 1
                    if not cfg['chal_fixed']:
                        chal = p_meas.copy()
                else:
                    n_chal = 1
                    chal = p_meas.copy()
                if n_chal >= cfg['chal_n']:
                    X = p_meas.copy(); P = Rm_reset + np.diag([0.1 ** 2, 0.1 ** 2, 0.003 ** 2])
                    info['status'] = 'reset'; n_rej = 0; last_fix = stamp; chal, n_chal = None, 0
        # output at the image stamp: propagate from t + dt to t with the current speed / yaw rate
        v, wz = cs.at(stamp)
        h = -dt_ns / 1e9
        xo, yo, yo_aw = X[0] + v * h * np.cos(X[2]), X[1] + v * h * np.sin(X[2]), wrap(X[2] + wz * h)
        torch.cuda.synchronize(); times.append((time.perf_counter() - tic) * 1000)
        if cfg['log_sim'] and reloc is not None and 'ml_sim_top1' not in info:
            reloc.candidates(list(sets.iloc[k][cams]), prior=X)       # report only, after the timer
            info.update(reloc.last)
        rows.append((stamp, xo, yo, yo_aw))
        info.update(ms=times[-1], sx=np.sqrt(P[0, 0]), sy=np.sqrt(P[1, 1]), syaw=np.sqrt(P[2, 2]))
        infos.append(info)
        if verbose and k % 200 == 0:
            err = '' if qrun is not None else 'err %.3f' % np.hypot(xo - data.interp_gt([stamp])[0][0], yo - data.interp_gt([stamp])[1][0])
            print(k, len(sets), f'{np.mean(times[1:]) if len(times) > 1 else 0:.1f} ms', info.get('status'),
                  info.get('n_m1'), info.get('n_inl'), err, flush=True)
    pred = pd.DataFrame(rows, columns=['timestamp_ns', 'x', 'y', 'yaw'])
    pred.to_csv(f'{out}/pred.csv', index=False, float_format='%.4f')
    inf = pd.DataFrame(infos); inf.insert(0, 'timestamp_ns', pred.timestamp_ns); inf.to_csv(f'{out}/online_info.csv', index=False)
    ms = float(np.mean(times[1:]))
    stat = inf.status.value_counts(normalize=True).mul(100).round(2).to_dict()
    json.dump(dict(cfg=cfg, cams=cams, split=split), open(f'{out}/config.json', 'w'), indent=1)
    return pred, inf, ms, stat


def evaluate(out, split, name, ms, stat, video=False, baselines=None, others=None, extra=None):
    ex = dict(method='EKF: car-state dead reckoning + map 3-DoF PnP fixes (guided matching)', runtime_ms_per_frame=ms,
              **{f'status_{k}_pct': v for k, v in stat.items()}, **(extra or {}))
    m = report.make_report(f'{out}/pred.csv', name, split, video, ex, baselines=baselines, others=others)
    return m


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='tune'); ap.add_argument('--out', required=True)
    ap.add_argument('--map', default='outputs/20_map'); ap.add_argument('--cams', nargs='+', default=data.CAMS)
    ap.add_argument('--limit', type=int); ap.add_argument('--name', default='online localiser')
    ap.add_argument('--video', action='store_true'); ap.add_argument('--cfg', default='{}', help='json overrides')
    ap.add_argument('--query', help='extracted run to localise (split test): images/, state.csv, calib.json, first pose')
    ap.add_argument('--map_until', type=float, help='test mode: use only map frames before this time [s] (plumbing checks)')
    ap.add_argument('--fold', choices=list(data.FOLDS), help='cross-validation fold: --split is a lap (L0..L8)')
    fp = ap.add_mutually_exclusive_group()
    fp.add_argument('--first_pose', metavar='FILE', help='test mode: the given first pose of the run (csv, first row: '
                    'timestamp_ns,x,y,yaw; yaw in rad; without timestamp_ns = the pose at the first image)')
    fp.add_argument('--no_first_pose', action='store_true', help='test mode: no first pose given; estimate a stand-in '
                    'from the car-state motion matched to the map run (odo.first_pose, a few m off; start sigma 3 m / 1 deg)')
    a = ap.parse_args()
    cfg = json.loads(a.cfg)
    if a.query:
        if a.first_pose:
            cfg['init_pose'] = list(data.Run(a.query).first_pose(a.first_pose))
        elif a.no_first_pose:
            from .first_pose import stand_in
            pose, sinfo = stand_in(a.query)
            cfg['init_pose'] = list(pose)
            cfg.setdefault('init_sigma_xy', 3.0); cfg.setdefault('init_sigma_yaw_deg', 1.0)
            os.makedirs(a.out, exist_ok=True)
            json.dump(sinfo, open(f'{a.out}/first_pose_standin.json', 'w'))
            print(f'stand-in first pose ({pose[1]:.2f}, {pose[2]:.2f}, {np.degrees(pose[3]):.2f} deg): {sinfo["method"]}')
        else:
            ap.exit(2, 'odo.online: test mode (--query) needs a start-pose flag:\n'
                       '  --first_pose FILE   the given first pose of the run (csv: timestamp_ns,x,y,yaw; first row)\n'
                       '  --no_first_pose     no first pose given: stand-in from the car-state motion (a few m off)\n')
    elif a.first_pose or a.no_first_pose:
        ap.error('--first_pose / --no_first_pose are for test mode (--query); research splits start at their GT pose')
    pred, inf, ms, stat = run(a.split, a.out, a.map, a.cams, a.limit, cfg, query_root=a.query,
                              map_until_s=a.map_until, fold=a.fold)
    if a.query:
        print(f'{len(pred)} poses written to {a.out}/pred.csv, {ms:.1f} ms/frame', stat)
        raise SystemExit
    m = evaluate(a.out, a.split, a.name, ms, stat, a.video,
                 baselines={'Exp12 PnP (no filter)': 'outputs/12_reloc_pnp/val/metrics_val.json'} if a.split == 'val' else None)
    print({k: round(v, 4) for k, v in m.items() if isinstance(v, float)}, stat, f'{ms:.1f} ms/frame')
