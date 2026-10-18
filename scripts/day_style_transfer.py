"""Day proxy by photorealistic style transfer: CAP-VSTNet (CVPR 2023, reversible network + cWCT, photorealistic video
checkpoint). The content (geometry, structures, textures) of every night frame is kept pixel-aligned; only its feature
statistics are re-coloured towards a daylight reference image of the same camera. Light sources (pixels >= 200) are
removed first (as day_nolights), because lamps are off in daylight and style transfer alone keeps them bright.

Style reference per camera: the first image in <style_dir>/images/<cam>/ (we use one Qwen-Image-Edit day render of the
same camera, scripts/day_translate_qwen.py; only its colour / tone statistics are used).

    python scripts/day_style_transfer.py --lap L2 --cams cl cr --style_dir data/day_proxy/style_refs_raw \
        --out data/day_proxy/L2_style            (env daytrans, GPU, in tmux)
Output layout as scripts/day_translate.py (<out>/images/<cam>/<file>.jpg + <cam>.csv), read as images 'day_style'.
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
sys.path.insert(0, f'{ROOT}/CAP-VSTNet'); sys.path.insert(0, f'{ROOT}/scripts')
from day_translate import fill_holes  # noqa: E402  (same light removal as day_nolights)

DATA = f'{ROOT}/data/extracted'
LAP_STARTS = [0.0, 185.456, 365.569, 523.375, 633.972, 752.192, 896.116, 1007.673, 1120.703, 1194.393]


def to_tensor(rgb):
    return torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).cuda()


def pad_to(x, k):
    """Reflect-pad (B,C,H,W) to multiples of k; returns padded tensor and the original size."""
    H, W = x.shape[-2:]
    return torch.nn.functional.pad(x, (0, (-W) % k, 0, (-H) % k), mode='reflect'), (H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lap', default='L2'); ap.add_argument('--out', required=True)
    ap.add_argument('--cams', nargs='+', default=['cl', 'cr']); ap.add_argument('--style_dir', required=True)
    ap.add_argument('--limit', type=int); ap.add_argument('--t0', type=float)
    ap.add_argument('--ckpt', default=f'{ROOT}/CAP-VSTNet/checkpoints/checkpoints/photo_video.pt')
    ap.add_argument('--keep_lights', action='store_true')
    a = ap.parse_args()
    from models.RevResNet import RevResNet
    from models.cWCT import cWCT
    net = RevResNet(nBlocks=[10, 10, 10], nStrides=[1, 2, 2], nChannels=[16, 64, 256], in_channel=3, mult=4,
                    hidden_dim=16, sp_steps=2)
    net.load_state_dict(torch.load(a.ckpt)['state_dict']); net = net.cuda().eval()
    cwct = cWCT()
    k = net.down_scale
    gt0 = pd.read_csv(f'{DATA}/gt.csv', nrows=1).timestamp_ns.iloc[0]
    i = int(a.lap[1:]); ta, tb = LAP_STARTS[i], LAP_STARTS[i + 1]
    if a.t0 is not None:
        ta = LAP_STARTS[i] + a.t0
    pool = ThreadPoolExecutor(8)
    for cam in a.cams:
        sd = f'{a.style_dir}/images/{cam}'
        sf = sorted(f for f in os.listdir(sd) if f.endswith('.jpg'))[0]
        style = cv2.cvtColor(cv2.imread(f'{sd}/{sf}'), cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            z_s = net(pad_to(to_tensor(style), k)[0], forward=True)
        f = pd.read_csv(f'{DATA}/images/{cam}.csv')
        t = (f.stamp_ns - gt0) / 1e9
        f = f[(t >= ta - 0.5) & (t < tb + 0.5)].reset_index(drop=True)
        if a.limit:
            f = f.iloc[:a.limit]
        os.makedirs(f'{a.out}/images/{cam}', exist_ok=True)
        todo = [r for r in f.itertuples() if not os.path.exists(f'{a.out}/{r.file}')]
        tic = time.time(); writes = []
        for n, (r, im) in enumerate(zip(todo, pool.map(lambda r: cv2.imread(f'{DATA}/{r.file}'), todo))):
            x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if not a.keep_lights:
                mask = cv2.dilate((x.max(2) >= 200).astype(np.uint8), np.ones((13, 13), np.uint8))
                x = fill_holes(x, mask)
            with torch.no_grad():
                xc, (H, W) = pad_to(to_tensor(x), k)
                y = net(cwct.transfer(net(xc, forward=True), z_s, None, None), forward=False)[..., :H, :W]
            y = (y[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
            writes.append(pool.submit(cv2.imwrite, f'{a.out}/{r.file}', cv2.cvtColor(y, cv2.COLOR_RGB2BGR),
                                      [cv2.IMWRITE_JPEG_QUALITY, 95]))
            if n % 200 == 0:
                print(cam, n, len(todo), f'{(time.time() - tic) / (n + 1) * 1000:.0f} ms/img', flush=True)
        for w_ in writes:
            w_.result()
        f.to_csv(f'{a.out}/images/{cam}.csv', index=False)
        print(cam, 'done', len(todo), f'{time.time() - tic:.0f} s', 'style', sf, flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
