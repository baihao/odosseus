"""Stage 3 (offline): a cached 3D feature map of the training run (method 3a of the brief).

For every training frame A of every camera: SuperPoint keypoints (car body cropped), matched with LightGlue to the
previous and the next frame of the same camera with >= `min_base` m baseline, triangulated with the training GT
6-DoF poses (looked up at stamp + dt, the image-vs-GT clock offset fitted in Exp12). Points seen in both pairs keep
the wider-angle estimate if the two agree (else they are dropped). Only triangulated keypoints are stored:
pixel position in A (raw, distorted), 256-D descriptor (fp16) and 3D point in the map frame.

The map is built over the whole run once; `odo.online` filters it by time (map frames AND partners must lie in the
map split, e.g. t < 897 s for validation, everything for the hidden test run).

    python -m odo.map_build --cams fl fr cl cr r rl rr --out outputs/20_map
Writes <out>/<cam>.npz: frame table (stamp, partner stamps, base pose, offset, count) + concatenated points.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from . import data
from .feature_vo import Extractor, load_gray
from .prep import CROP_BOTTOM
from .reloc_pnp import GT6, triangulate, undist

DT_NS = -9_100_000  # image stamp vs GT clock (Exp12, tune lap)


def crop_feats(f, cam, H):
    keep = f['keypoints'][0, :, 1] < CROP_BOTTOM[cam] * H
    return {k: v[:, keep] if v.dim() > 1 and v.shape[1] == keep.shape[0] else v for k, v in f.items()}


def partner(pos, a, step, min_base, max_k=10):
    for k in range(1, max_k + 1):
        b = a + step * k
        if b < 0 or b >= len(pos):
            return None
        if np.linalg.norm(pos[b] - pos[a]) >= min_base:
            return b
    return None


def build_cam(cam, ext, gt6, out, min_base=1.5, max_kp=2048):
    fr = data.frames(cam)
    st = fr.stamp_ns.values
    c = data.calib()[cam]; K, D, Tbc = c['K'], c['D'], c['T_base_cam']
    H = c['height']; f_px = 0.5 * (K[0, 0] + K[1, 1])
    Tb = gt6.T(st + DT_NS)                     # base poses (GT, map = training run)
    Tc = Tb @ Tbc
    pos = Tb[:, :3, 3]
    N = len(st)
    pool = ThreadPoolExecutor(8)
    load = lambda i: load_gray(fr.path[i], 1.0)
    futs = {i: pool.submit(load, i) for i in range(min(24, N))}
    cache = {}

    def feats(i):
        if i not in cache:
            if i not in futs:
                futs[i] = pool.submit(load, i)
            cache[i] = crop_feats(ext.features(futs.pop(i).result()), cam, H)
            for j in range(i + 1, min(i + 24, N)):
                if j not in futs and j not in cache:
                    futs[j] = pool.submit(load, j)
        return cache[i]

    def tri(fa, a, b):
        fb = feats(b)
        m = ext.matcher({'image0': fa, 'image1': fb})['matches'][0].cpu().numpy()
        kA = fa['keypoints'][0].cpu().numpy(); kB = fb['keypoints'][0].cpu().numpy()
        if len(m) < 8:
            return np.full((kA.shape[0], 3), np.nan), np.zeros(kA.shape[0])
        X, ok = triangulate(undist(kA[m[:, 0]], K, D), undist(kB[m[:, 1]], K, D), Tc[a], Tc[b], max_reproj=2.0 / f_px)
        ra = X - Tc[a][:3, 3]; rb = X - Tc[b][:3, 3]
        ang = np.arccos(np.clip((ra * rb).sum(1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1) + 1e-12), -1, 1))
        Xf = np.full((kA.shape[0], 3), np.nan); Af = np.zeros(kA.shape[0])
        Xf[m[ok, 0]] = X[ok]; Af[m[ok, 0]] = ang[ok]
        return Xf, Af

    tab = dict(stamp=[], stamp_prev=[], stamp_next=[], offset=[], count=[], T=[])
    KP, DE, XX = [], [], []
    off = 0; t0 = time.time()
    with torch.no_grad():
        for a in range(N):
            bn, bp = partner(pos, a, +1, min_base), partner(pos, a, -1, min_base)
            if bn is None and bp is None:
                cache.pop(a - 12, None)
                continue
            fa = feats(a)
            Xs = [tri(fa, a, b) if b is not None else None for b in (bp, bn)]
            Xs = [x for x in Xs if x is not None]
            if len(Xs) == 2:
                (X1, A1), (X2, A2) = Xs
                both = np.isfinite(X1[:, 0]) & np.isfinite(X2[:, 0])
                depth = np.linalg.norm(np.where(np.isfinite(X1), X1, X2) - Tc[a][:3, 3], axis=1)
                agree = np.linalg.norm(np.nan_to_num(X1 - X2), axis=1) < 0.05 * depth + 0.1
                X = np.where((A2 > A1)[:, None], X2, X1)
                X = np.where(np.isfinite(X), X, np.where(np.isfinite(X1), X1, X2))
                X[both & ~agree] = np.nan
            else:
                X = Xs[0][0]
            v = np.isfinite(X[:, 0])
            kp = fa['keypoints'][0].cpu().numpy()[v]
            de = fa['descriptors'][0].half().cpu().numpy()[v]
            KP.append(kp.astype(np.float32)); DE.append(de); XX.append(X[v].astype(np.float32))
            tab['stamp'].append(st[a]); tab['stamp_prev'].append(st[bp] if bp is not None else 0)
            tab['stamp_next'].append(st[bn] if bn is not None else 0)
            tab['offset'].append(off); tab['count'].append(int(v.sum())); tab['T'].append(Tb[a])
            off += int(v.sum())
            for j in [k for k in cache if k < a - 12]:
                cache.pop(j)
            if a % 500 == 0:
                print(cam, a, N, f'{(time.time() - t0) / (a + 1) * 1000:.0f} ms/frame', 'pts', int(v.sum()), flush=True)
    np.savez(f'{out}/{cam}.npz', kp=np.concatenate(KP), desc=np.concatenate(DE), X=np.concatenate(XX),
             **{k: np.array(v) for k, v in tab.items()}, dt_ns=DT_NS, min_base=min_base, max_kp=max_kp)
    print(cam, 'done', len(tab['stamp']), 'frames', off, 'points', f'{(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cams', nargs='+', default=data.CAMS); ap.add_argument('--out', required=True)
    ap.add_argument('--min_base', type=float, default=1.5); ap.add_argument('--max_kp', type=int, default=2048)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ext = Extractor('superpoint', a.max_kp)
    gt6 = GT6()
    for cam in a.cams:
        build_cam(cam, ext, gt6, a.out, a.min_base, a.max_kp)
