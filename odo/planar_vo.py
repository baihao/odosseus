"""Planar, Ackermann-constrained, metric-step VO with known extrinsics (1 DoF per step).

The car moves on the track plane with negligible sideslip, so between two frames base_link undergoes
    R_b = Rz(theta),   t_b = d * [cos(theta/2), sin(theta/2), 0]      (circular-arc chord)
with d = distance from integrated car-state speed (allowed input). Through the known extrinsics this
fixes the full relative pose of every camera (lever arm included) as a function of theta alone:
    R_c = Rbc^T Rz Rbc,    t_c = Rbc^T (Rz tbc + t_b - tbc),    E = [t_c]x R_c
theta is found per camera and step by a robust (truncated-Sampson, MSAC) 1-D search over a grid,
refined on a fine grid. This replaces the ill-conditioned 5-DoF essential matrix.
Optionally a second DoF beta (chord direction offset, sideslip) is searched too.

    python -m odo.planar_vo --matches outputs/05_planar_vo/steps_val --out outputs/05_planar_vo/theta_val
Writes <out>/<cam>.npz with t_prev, t_cur, dyaw (theta), ok, n_inl, cost_curv, like odo.feature_vo
so that odo.fuse_vo can fuse it.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import data, vo2map


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def essentials(thetas, d, Rbc, tbc, beta=0.0):
    G = len(thetas)
    c, s = np.cos(thetas), np.sin(thetas)
    Rz = np.zeros((G, 3, 3)); Rz[:, 0, 0] = c; Rz[:, 0, 1] = -s; Rz[:, 1, 0] = s; Rz[:, 1, 1] = c; Rz[:, 2, 2] = 1
    tb = d * np.stack([np.cos(thetas / 2 + beta), np.sin(thetas / 2 + beta), np.zeros(G)], 1)
    Rc = np.einsum('ji,gjk,kl->gil', Rbc, Rz, Rbc)
    tc = np.einsum('ji,gj->gi', Rbc, np.einsum('gij,j->gi', Rz, tbc) + tb - tbc)
    S = np.zeros((G, 3, 3))
    S[:, 0, 1] = -tc[:, 2]; S[:, 0, 2] = tc[:, 1]; S[:, 1, 0] = tc[:, 2]
    S[:, 1, 2] = -tc[:, 0]; S[:, 2, 0] = -tc[:, 1]; S[:, 2, 1] = tc[:, 0]
    return S @ Rc


def sampson(E, f0, f1):
    """E (G,3,3); f0,f1 (N,3) homogeneous normalized coords, constraint f0^T E f1 = 0. -> (G,N)"""
    Ef1 = np.einsum('gij,nj->gni', E, f1)
    Etf0 = np.einsum('gji,nj->gni', E, f0)
    num = np.einsum('ni,gni->gn', f0, Ef1) ** 2
    den = Ef1[..., 0] ** 2 + Ef1[..., 1] ** 2 + Etf0[..., 0] ** 2 + Etf0[..., 1] ** 2
    return num / np.maximum(den, 1e-18)


def solve_step(m, d, Rbc, tbc, f_px, thr_px=1.5, coarse=np.radians(np.arange(-8, 8.001, 0.05)), beta=0.0):
    if len(m) < 10 or d < 0.2:
        return np.nan, 0, np.nan
    f0 = np.c_[m[:, 0:2], np.ones(len(m))]; f1 = np.c_[m[:, 2:4], np.ones(len(m))]
    t2 = (thr_px / f_px) ** 2

    def cost(th):
        e = sampson(essentials(th, d, Rbc, tbc, beta), f0, f1)
        return np.minimum(e / t2, 1.0).sum(1), e
    c, _ = cost(coarse)
    th0 = coarse[np.argmin(c)]
    fine = th0 + np.radians(np.arange(-0.08, 0.0801, 0.002))
    cf, e = cost(fine)
    j = np.argmin(cf)
    # parabolic sub-grid refinement + curvature (confidence)
    if 0 < j < len(fine) - 1:
        a, b, cc = cf[j - 1], cf[j], cf[j + 1]
        den = a - 2 * b + cc
        off = 0.5 * (a - cc) / den if den > 0 else 0.0
        curv = den / (fine[1] - fine[0]) ** 2
    else:
        off, curv = 0.0, 0.0
    th = fine[j] + off * (fine[1] - fine[0])
    n_inl = int((e[j] < t2).sum())
    return th, n_inl, curv


def run_camera(args):
    steps_dir, cam, out, thr_px = args
    z = np.load(f'{steps_dir}/{cam}.npz'); mm = np.load(f'{steps_dir}/{cam}_matches.npz')
    T = data.calib()[cam]['T_base_cam']; Rbc, tbc = T[:3, :3], T[:3, 3]
    K = data.calib()[cam]['K']; f_px = 0.5 * (K[0, 0] + K[1, 1])
    t_prev, t_cur = z['t_prev'], z['t_cur']
    d = np.array([vo2map.speed_dist([a, b])[0] for a, b in zip(t_prev, t_cur)])
    off, M = mm['offsets'], mm['m']
    N = len(t_prev)
    th = np.full(N, np.nan); ni = np.zeros(N, int); cv = np.zeros(N)
    t0 = time.time()
    for k in range(N):
        th[k], ni[k], cv[k] = solve_step(M[off[k]:off[k + 1]].astype(np.float64), d[k], Rbc, tbc, f_px, thr_px)
    ok = np.isfinite(th) & (ni >= 20)
    os.makedirs(out, exist_ok=True)
    R = np.tile(np.eye(3), (N, 1, 1))
    np.savez(f'{out}/{cam}.npz', t_prev=t_prev, t_cur=t_cur, dyaw_base=np.nan_to_num(th), ok=ok, n_inl=ni,
             curv=cv, R=R, t=np.zeros((N, 3)), n_match=np.diff(off), ms_per_frame=float(z['ms_per_frame']),
             solve_ms=(time.time() - t0) / N * 1000)
    return cam, float(ok.mean()), float(np.median(ni)), (time.time() - t0) / N * 1000


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--matches', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--cams', nargs='+', default=data.CAMS); ap.add_argument('--thr_px', type=float, default=1.5)
    a = ap.parse_args()
    with ProcessPoolExecutor(7) as ex:
        for r in ex.map(run_camera, [(a.matches, c, a.out, a.thr_px) for c in a.cams]):
            print(r, flush=True)
