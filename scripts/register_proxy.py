"""Register a generated day proxy back onto the geometry of its night source frame.

Image editors (Qwen-Image-Edit-2509) slightly zoom / shift their output (median 7.9 px on our frames). That would be a
fake pose error, not an appearance change. For every frame: SuperPoint + LightGlue between the night original and the
raw day image, one homography (RANSAC, 3 px), and the day image is warped into the night frame's pixel grid. Only this
global transform is removed; the generator's local warping (~2.8 px median) stays in, which makes the proxy slightly
pessimistic. Frames with < 30 homography inliers are copied unwarped and listed in <out>/register_log.csv.

    python scripts/register_proxy.py data/day_proxy/L2_qwen_raw data/day_proxy/L2_qwen   (env mast3r-slam, GPU)
Can run while the translation is still writing: it processes the frames present and skips frames already registered;
run it again at the end."""
import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

src, dst = sys.argv[1], sys.argv[2]
sp = SuperPoint(max_num_keypoints=2048).eval().cuda(); lg = LightGlue(features='superpoint').eval().cuda()
log = []
for cam in ['fl', 'fr', 'cl', 'cr', 'r', 'rl', 'rr']:
    if not os.path.isdir(f'{src}/images/{cam}'):
        continue
    os.makedirs(f'{dst}/images/{cam}', exist_ok=True)
    for f in sorted(os.listdir(f'{src}/images/{cam}')):
        if not f.endswith('.jpg') or os.path.exists(f'{dst}/images/{cam}/{f}'):
            continue
        night = f'data/extracted/images/{cam}/{f}'
        with torch.no_grad():
            f0 = sp.extract(load_image(night).cuda(), resize=1024)
            f1 = sp.extract(load_image(f'{src}/images/{cam}/{f}').cuda(), resize=1024)
            m = rbd(lg({'image0': f0, 'image1': f1}))['matches'].cpu().numpy()
        k0 = rbd(f0)['keypoints'].cpu().numpy()[m[:, 0]]; k1 = rbd(f1)['keypoints'].cpu().numpy()[m[:, 1]]
        H, n_inl = None, 0
        if len(m) >= 8:
            H, inl = cv2.findHomography(k1, k0, cv2.RANSAC, 3.0)
            n_inl = int(inl.sum()) if H is not None else 0
        im = cv2.imread(f'{src}/images/{cam}/{f}')
        if H is not None and n_inl >= 30:
            im = cv2.warpPerspective(im, H, (im.shape[1], im.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(f'{dst}/images/{cam}/{f}', im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        log.append(dict(cam=cam, file=f, matches=len(m), inliers=n_inl, warped=H is not None and n_inl >= 30,
                        shift_px=float(np.median(np.linalg.norm(k0 - k1, axis=1))) if len(m) else np.nan))
    if os.path.exists(f'{src}/images/{cam}.csv'):
        shutil.copy(f'{src}/images/{cam}.csv', f'{dst}/images/{cam}.csv')
    print(cam, 'registered', flush=True)
if log:
    lf = f'{dst}/register_log.csv'
    pd.concat(([pd.read_csv(lf)] if os.path.exists(lf) else []) + [pd.DataFrame(log)]).to_csv(lf, index=False)
    d = pd.DataFrame(log)
    print(f'{len(d)} frames, warped {d.warped.mean() * 100:.1f} %, median raw shift {d.shift_px.median():.2f} px, '
          f'median inliers {d.inliers.median():.0f}', flush=True)
print('REGISTER_DONE', flush=True)
