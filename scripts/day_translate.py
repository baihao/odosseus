"""Day proxy of the night query images: CycleGAN-Turbo `night_to_day` (img2img-turbo, trained on BDD100k driving
scenes) applied to every camera image of a lap. Env `daytrans` (GPU, run in tmux).

The model runs at the resolution SuperPoint sees online (long side 1024, padded to a multiple of 64), the output is
resized back to the original 1506x728 and written as JPEG under <out>/images/<cam>/ with the original file names,
plus <out>/images/<cam>.csv (same layout as data/extracted). odo.online reads it with cfg img_root=<out>.

--nolights: light sources (pixels >= 200, dilated) are inpainted before the translation, so lamps and lit windows,
which are dark in daylight, give no features (the translation alone keeps them glowing).

    python scripts/day_translate.py --lap L2 --out data/day_proxy/L2 [--nolights] [--limit N]
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{ROOT}/img2img-turbo/src')
DATA = f'{ROOT}/data/extracted'
CAMS = ['fl', 'fr', 'cl', 'cr', 'r', 'rl', 'rr']
LAP_STARTS = [0.0, 185.456, 365.569, 523.375, 633.972, 752.192, 896.116, 1007.673, 1120.703, 1194.393]


def fill_holes(x, mask):
    """Fill masked pixels with a normalised Gaussian blur of the unmasked ones (fast stand-in for inpainting;
    the blur is computed at 1/4 resolution)."""
    H, W = mask.shape
    w = (1 - mask).astype(np.float32)
    xs = cv2.resize(x.astype(np.float32) * w[..., None], (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    ws = cv2.resize(w, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    out = x.astype(np.float32)
    todo = mask > 0
    for s in (2, 6, 20):
        num = cv2.resize(cv2.GaussianBlur(xs, (0, 0), s), (W, H))
        den = cv2.resize(cv2.GaussianBlur(ws, (0, 0), s), (W, H))[..., None]
        sel = todo & (den[..., 0] > 0.05)
        out[sel] = (num / np.maximum(den, 1e-3))[sel]
        todo &= ~sel
    return out.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lap', default='L2'); ap.add_argument('--out', required=True)
    ap.add_argument('--cams', nargs='+', default=CAMS); ap.add_argument('--limit', type=int)
    ap.add_argument('--t0', type=float, help='start offset into the lap [s]'); ap.add_argument('--t1', type=float)
    ap.add_argument('--long_side', type=int, default=1024); ap.add_argument('--nolights', action='store_true')
    a = ap.parse_args()
    from cyclegan_turbo import CycleGAN_Turbo
    model = CycleGAN_Turbo(pretrained_name='night_to_day', ckpt_folder=f'{ROOT}/img2img-turbo/checkpoints')
    model.eval(); model.half()
    gt0 = pd.read_csv(f'{DATA}/gt.csv', nrows=1).timestamp_ns.iloc[0]
    i = int(a.lap[1:])
    ta, tb = LAP_STARTS[i], LAP_STARTS[i + 1]
    if a.t0 is not None:
        ta, tb = LAP_STARTS[i] + a.t0, LAP_STARTS[i] + (a.t1 if a.t1 is not None else tb - LAP_STARTS[i])
    pool = ThreadPoolExecutor(8)
    for cam in a.cams:
        f = pd.read_csv(f'{DATA}/images/{cam}.csv')
        t = (f.stamp_ns - gt0) / 1e9
        f = f[(t >= ta - 0.5) & (t < tb + 0.5)].reset_index(drop=True)
        if a.limit:
            f = f.iloc[:a.limit]
        os.makedirs(f'{a.out}/images/{cam}', exist_ok=True)
        todo = [r for r in f.itertuples() if not os.path.exists(f'{a.out}/{r.file}')]
        tic = time.time()
        loads = pool.map(lambda r: cv2.imread(f'{DATA}/{r.file}'), todo)
        writes = []
        for n, (r, im) in enumerate(zip(todo, loads)):
            H, W = im.shape[:2]
            w = a.long_side; h = int(round(H * w / W))
            hp = (h + 63) // 64 * 64
            x = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_AREA)
            if a.nolights:
                mask = cv2.dilate((x.max(2) >= 200).astype(np.uint8), np.ones((9, 9), np.uint8))
                x = fill_holes(x, mask)
            x = np.pad(x, ((0, hp - h), (0, 0), (0, 0)), mode='reflect')
            xt = torch.from_numpy(x).permute(2, 0, 1)[None].cuda().half().div(127.5).sub(1)
            with torch.no_grad():
                y = model(xt, direction=None, caption=None)
            y = ((y[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()[:h]
            y = cv2.resize(y, (W, H), interpolation=cv2.INTER_CUBIC)
            writes.append(pool.submit(cv2.imwrite, f'{a.out}/{r.file}', cv2.cvtColor(y, cv2.COLOR_RGB2BGR),
                                      [cv2.IMWRITE_JPEG_QUALITY, 95]))
            if n % 200 == 0:
                print(cam, n, len(todo), f'{(time.time() - tic) / (n + 1) * 1000:.0f} ms/img', flush=True)
        for w_ in writes:
            w_.result()
        f.to_csv(f'{a.out}/images/{cam}.csv', index=False)
        print(cam, 'done', len(todo), f'{time.time() - tic:.0f} s', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
