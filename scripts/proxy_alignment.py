"""Geometric fidelity of a day proxy: SuperPoint + LightGlue between each night frame and its day version (same pose).
A faithful proxy keeps matched points in place (displacement ~0 px); the number of matches measures the appearance gap.
    python scripts/proxy_alignment.py <proxy_dir> [<proxy_dir> ...] [--n 30]
Uses the files present in the first proxy folder (every k-th, up to n per camera)."""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, '.')
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

ap = argparse.ArgumentParser(); ap.add_argument('dirs', nargs='+'); ap.add_argument('--n', type=int, default=30)
a = ap.parse_args()
sp = SuperPoint(max_num_keypoints=2048).eval().cuda(); lg = LightGlue(features='superpoint').eval().cuda()
for d in a.dirs:
    res = []
    for cam in ['fl', 'fr', 'cl', 'cr', 'r', 'rl', 'rr']:
        p = f'{a.dirs[0]}/images/{cam}'
        if not os.path.isdir(p):
            continue
        fs = sorted(f for f in os.listdir(p) if f.endswith('.jpg'))
        for f in fs[::max(1, len(fs) // a.n)][:a.n]:
            if not os.path.exists(f'{d}/images/{cam}/{f}'):
                continue
            with torch.no_grad():
                f0 = sp.extract(load_image(f'data/extracted/images/{cam}/{f}').cuda(), resize=1024)
                f1 = sp.extract(load_image(f'{d}/images/{cam}/{f}').cuda(), resize=1024)
                m = rbd(lg({'image0': f0, 'image1': f1}))['matches'].cpu().numpy()
            k0 = rbd(f0)['keypoints'].cpu().numpy()[m[:, 0]]; k1 = rbd(f1)['keypoints'].cpu().numpy()[m[:, 1]]
            dist = np.linalg.norm(k0 - k1, axis=1)
            # residual after one global homography (what registration to the source frame could remove)
            hres = np.nan
            if len(m) >= 8:
                Hm, inl = cv2.findHomography(k1, k0, cv2.RANSAC, 3.0)
                if Hm is not None:
                    pr = cv2.perspectiveTransform(k1.reshape(-1, 1, 2), Hm).reshape(-1, 2)
                    hres = np.median(np.linalg.norm(pr - k0, axis=1))
            res.append((cam, len(m), np.median(dist) if len(m) else np.nan, np.mean(dist < 3) * 100 if len(m) else np.nan, hres))
    r = np.array([x[1:] for x in res], float)
    print(f'{d}: {len(res)} frames, matches night<->proxy median {np.median(r[:, 0]):.0f}, displacement median '
          f'{np.nanmedian(r[:, 1]):.2f} px, share of matches within 3 px {np.nanmedian(r[:, 2]):.1f} %, '
          f'median residual after a global homography {np.nanmedian(r[:, 3]):.2f} px', flush=True)
print('ALIGN_DONE')
