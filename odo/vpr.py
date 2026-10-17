"""Global descriptors for visual place recognition (stage 2, method 3b of the brief).

    python -m odo.vpr --model megaloc --cams fl fr r --splits train val --stride_db 1 --out outputs/10_vpr/desc

Models
  megaloc : MegaLoc (DINOv2-B + SALAD aggregation trained on a union of VPR datasets), 8448-D
  dinov2  : DINOv2 ViT-B/14 patch tokens, GeM pooled + CLS (simple foundation-model baseline), 1536-D
Images: car body cropped (prep.CROP_BOTTOM), resized to height `--h` (multiple of 14), ImageNet-normalised.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import data
from .prep import CROP_BOTTOM

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_model(name):
    if name == 'megaloc':
        m = torch.hub.load('gmberton/MegaLoc', 'get_trained_model', trust_repo=True)
    elif name == 'dinov2':
        bb = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', trust_repo=True)

        class G(torch.nn.Module):
            def __init__(s):
                super().__init__(); s.bb = bb

            def forward(s, x):
                o = s.bb.forward_features(x)
                p = o['x_norm_patchtokens'].clamp(min=1e-6)
                gem = p.pow(3).mean(1).pow(1 / 3)
                return F.normalize(torch.cat([F.normalize(o['x_norm_clstoken'], dim=-1), F.normalize(gem, dim=-1)], -1), dim=-1)
        m = G()
    else:
        raise ValueError(name)
    return m.eval().cuda()


def load_img(path, cam, h):
    im = cv2.imread(path)[:, :, ::-1]
    H, W = im.shape[:2]
    im = im[:int(round(H * CROP_BOTTOM[cam]))]
    w = int(round(im.shape[1] * h / im.shape[0] / 14)) * 14
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)


@torch.no_grad()
def describe(model, paths, cam, h=322, bs=32):
    pool = ThreadPoolExecutor(16)
    out = []
    batches = [paths[i:i + bs] for i in range(0, len(paths), bs)]
    nxt = pool.map(lambda b: np.stack([load_img(p, cam, h) for p in b]), batches)
    for arr in nxt:
        x = torch.from_numpy(arr.copy()).permute(0, 3, 1, 2).float().div(255)
        x = ((x - MEAN) / STD).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            d = model(x)
        out.append(F.normalize(d.float(), dim=-1).cpu().numpy().astype(np.float16))
    return np.concatenate(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='megaloc'); ap.add_argument('--cams', nargs='+', default=['fl', 'fr', 'r'])
    ap.add_argument('--splits', nargs='+', default=['train', 'val']); ap.add_argument('--h', type=int, default=322)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = load_model(a.model)
    for split in a.splits:
        for cam in a.cams:
            fr = data.frames_split(cam, split)
            t0 = time.time()
            D = describe(model, list(fr.path), cam, a.h)
            ms = (time.time() - t0) / len(fr) * 1000
            np.savez(f'{a.out}/{a.model}_{cam}_{split}.npz', desc=D, stamp_ns=fr.stamp_ns.values, ms_per_img=ms)
            print(a.model, cam, split, D.shape, f'{ms:.1f} ms/img', flush=True)
