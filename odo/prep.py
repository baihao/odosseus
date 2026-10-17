"""Prepare per-camera image sequences for learned VO: undistort (plumb_bob), crop car body, downscale.

    python -m odo.prep --cam r --split val [--scale 0.5]

Writes data/prepared/<cam>_<split>/{000000.jpg..., stamps.csv, intrinsics.json, calib.yaml}
intrinsics.json: K of the written images (pinhole, no distortion), T_base_cam.
"""
import argparse
import json
import os

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from . import data

# rows kept (fraction of height) to drop the car body visible at the bottom of some cameras
CROP_BOTTOM = {'fl': 1.0, 'fr': 0.55, 'cl': 0.6, 'cr': 0.6, 'r': 1.0, 'rl': 1.0, 'rr': 0.6}


def undistorter(cam, scale):
    c = data.calib()[cam]
    W, H = c['width'], c['height']
    K, D = c['K'], c['D']
    Knew, _ = cv2.getOptimalNewCameraMatrix(K, D, (W, H), 0, (W, H), centerPrincipalPoint=False)
    mx, my = cv2.initUndistortRectifyMap(K, D, None, Knew, (W, H), cv2.CV_32FC1)
    h_keep = int(round(H * CROP_BOTTOM[cam]))
    Ks = Knew.copy(); Ks[:2] *= scale
    out_wh = (int(round(W * scale)), int(round(h_keep * scale)))

    def f(img):
        u = cv2.remap(img, mx, my, cv2.INTER_LINEAR)[:h_keep]
        return cv2.resize(u, out_wh, interpolation=cv2.INTER_AREA) if scale != 1 else u
    return f, Ks, out_wh


def prepare(cam, split, scale=0.5, stride=1, out_root=None):
    out = out_root or f'{data.ROOT}/data/prepared/{cam}_{split}' + (f'_s{stride}' if stride > 1 else '')
    os.makedirs(out, exist_ok=True)
    fr = data.frames_split(cam, split, stride)
    f, K, wh = undistorter(cam, scale)

    def work(k):
        p = f'{out}/{k:06d}.jpg'
        if not os.path.exists(p):
            cv2.imwrite(p, f(cv2.imread(fr.path[k])), [cv2.IMWRITE_JPEG_QUALITY, 95])
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(work, range(len(fr))))
    fr[['stamp_ns']].assign(file=[f'{k:06d}.jpg' for k in range(len(fr))]).to_csv(f'{out}/stamps.csv', index=False)
    json.dump(dict(K=K.tolist(), width=wh[0], height=wh[1], cam=cam,
                   T_base_cam=data.calib()[cam]['T_base_cam'].tolist()), open(f'{out}/intrinsics.json', 'w'), indent=1)
    with open(f'{out}/calib.yaml', 'w') as o:  # MASt3R-SLAM format
        o.write(f'width: {wh[0]}\nheight: {wh[1]}\n'
                f'calibration: [{K[0,0]:.4f}, {K[1,1]:.4f}, {K[0,2]:.4f}, {K[1,2]:.4f}]\n')
    print(out, len(fr), wh)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', nargs='+', default=['r']); ap.add_argument('--split', default='val')
    ap.add_argument('--scale', type=float, default=0.5); ap.add_argument('--stride', type=int, default=1)
    a = ap.parse_args()
    for c in a.cam:
        prepare(c, a.split, a.scale, a.stride)
