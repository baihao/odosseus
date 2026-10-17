"""Calibrated two-view feature VO per camera: SuperPoint/ALIKED + LightGlue matches between consecutive
frames, undistorted with the plumb_bob model, essential matrix (MAGSAC++) -> relative rotation and
translation direction. Causal: step k only uses frames k-1 and k.

    python -m odo.feature_vo --cams fl fr cl cr r rl rr --split val --out outputs/04_feature_vo/steps

Writes <out>/<cam>.npz: t_prev, t_cur (ns), R (N,3,3) and t (N,3) = motion of camera frame k expressed in
camera frame k-1 (t unit length), n_match, n_inl, ok.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from . import data
from .prep import CROP_BOTTOM


def load_gray(path, scale):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if scale != 1:
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return im


class Extractor:
    def __init__(self, feat='superpoint', max_kp=2048, device='cuda'):
        from lightglue import ALIKED, LightGlue, SuperPoint
        self.dev = device
        self.ext = (SuperPoint if feat == 'superpoint' else ALIKED)(max_num_keypoints=max_kp).eval().to(device)
        self.matcher = LightGlue(features=feat, depth_confidence=-1, width_confidence=-1).eval().to(device)

    @torch.no_grad()
    def features(self, im):
        t = torch.from_numpy(im).float().div(255)[None, None].to(self.dev)
        return self.ext.extract(t)

    @torch.no_grad()
    def match(self, f0, f1):
        m = self.matcher({'image0': f0, 'image1': f1})
        idx = m['matches'][0].cpu().numpy()
        k0 = f0['keypoints'][0].cpu().numpy()[idx[:, 0]]
        k1 = f1['keypoints'][0].cpu().numpy()[idx[:, 1]]
        return k0, k1


def two_view(k0, k1, K, D, thr_px=1.0):
    """k0,k1 pixel coords in the raw (distorted) image. Returns R, t (motion of cam1 in cam0 frame), inliers."""
    if len(k0) < 15:
        return None
    n0 = cv2.undistortPoints(k0.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)
    n1 = cv2.undistortPoints(k1.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)
    f = 0.5 * (K[0, 0] + K[1, 1])
    E, inl = cv2.findEssentialMat(n0, n1, np.eye(3), method=cv2.USAC_MAGSAC, prob=0.99999,
                                  threshold=thr_px / f, maxIters=5000)
    if E is None or E.shape != (3, 3):
        return None
    n_inl, R, t, inl2 = cv2.recoverPose(E, n0, n1, np.eye(3), mask=inl.copy())
    # cv2: x1 = R x0 + t  ->  camera-1 pose in camera-0 frame: R_rel = R^T, c = -R^T t
    return R.T, (-R.T @ t).ravel(), int(n_inl), len(k0)


def run_camera(cam, split, out, feat='superpoint', scale=1.0, max_kp=2048, stride=1, ext=None, save_matches=True):
    fr = data.frames_split(cam, split, stride)
    c = data.calib()[cam]
    K = c['K'].copy(); K[:2] *= scale
    D = c['D']
    H = int(c['height'] * scale)
    ymax = CROP_BOTTOM[cam] * H
    ext = ext or Extractor(feat, max_kp)
    N = len(fr)
    Rs = np.tile(np.eye(3), (N - 1, 1, 1)); ts = np.zeros((N - 1, 3)); nm = np.zeros(N - 1, int)
    ni = np.zeros(N - 1, int); ok = np.zeros(N - 1, bool)
    pool = ThreadPoolExecutor(8)
    futs = [pool.submit(load_gray, p, scale) for p in fr.path[:16]]
    prev = None; t0 = time.time(); matches = []
    for i in range(N):
        im = futs[i].result()
        if i + 16 < N:
            futs.append(pool.submit(load_gray, fr.path[i + 16], scale))
        f = ext.features(im)
        keep = f['keypoints'][0, :, 1] < ymax  # drop car-body keypoints
        f = {k: v[:, keep] if v.dim() > 1 and v.shape[1] == keep.shape[0] else v for k, v in f.items()}
        if prev is not None:
            k0, k1 = ext.match(prev, f)
            if save_matches and len(k0):
                n0 = cv2.undistortPoints(k0.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)
                n1 = cv2.undistortPoints(k1.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)
                matches.append(np.hstack([n0, n1]).astype(np.float32))
            elif save_matches:
                matches.append(np.zeros((0, 4), np.float32))
            r = two_view(k0, k1, K, D)
            if r is not None:
                Rs[i - 1], ts[i - 1], ni[i - 1], nm[i - 1] = r
                ok[i - 1] = ni[i - 1] >= 30
        prev = f
        if i % 500 == 0:
            print(cam, i, N, f'{(time.time() - t0) / max(i, 1) * 1000:.1f} ms/frame', flush=True)
    dt_ms = (time.time() - t0) / N * 1000
    os.makedirs(out, exist_ok=True)
    np.savez(f'{out}/{cam}.npz', t_prev=fr.stamp_ns.values[:-1], t_cur=fr.stamp_ns.values[1:], R=Rs, t=ts,
             n_match=nm, n_inl=ni, ok=ok, ms_per_frame=dt_ms)
    if save_matches:  # normalized (undistorted) image coords x0,y0,x1,y1 per step
        lens = np.array([len(m) for m in matches])
        np.savez(f'{out}/{cam}_matches.npz', offsets=np.concatenate([[0], np.cumsum(lens)]),
                 m=np.concatenate(matches) if matches else np.zeros((0, 4), np.float32))
    print(cam, 'done', f'{dt_ms:.1f} ms/frame', 'ok %.1f%%' % (100 * ok.mean()), 'median inliers', np.median(ni))
    return ext


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cams', nargs='+', default=data.CAMS); ap.add_argument('--split', default='val')
    ap.add_argument('--out', required=True); ap.add_argument('--feat', default='superpoint')
    ap.add_argument('--scale', type=float, default=1.0); ap.add_argument('--max_kp', type=int, default=2048)
    a = ap.parse_args()
    ext = None
    for cam in a.cams:
        ext = run_camera(cam, a.split, a.out, a.feat, a.scale, a.max_kp, ext=ext)
