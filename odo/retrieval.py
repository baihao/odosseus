"""Stage 2: global localisation of validation images by retrieval in the training run (map = training
images with their GT poses).

    python -m odo.retrieval --desc outputs/10_vpr/desc --model megaloc --cams fl fr r --out outputs/11_retrieval_megaloc

Per query timestamp (fl timeline, other cameras matched by nearest stamp) the cosine similarities of
each camera's descriptor to the database (training split, same camera) are averaged over cameras.
Pose = GT pose of the best database frame ('top1'), or a similarity-weighted mean over the top-k that lie
within `cluster_m` of the best one ('topk'). Each query uses only its own images -> causal.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from . import data, report


def aligned(desc_dir, model, cams, split, ref='fl'):
    """Descriptors of all cams resampled onto the ref-camera timeline (nearest stamp)."""
    z = np.load(f'{desc_dir}/{model}_{ref}_{split}.npz')
    ts = z['stamp_ns']
    D = []
    for c in cams:
        zc = np.load(f'{desc_dir}/{model}_{c}_{split}.npz')
        j = np.clip(np.searchsorted(zc['stamp_ns'], ts), 1, len(zc['stamp_ns']) - 1)
        j = np.where(np.abs(zc['stamp_ns'][j - 1] - ts) < np.abs(zc['stamp_ns'][j] - ts), j - 1, j)
        D.append(zc['desc'][j])
    return ts, D


def similarity(Dq, Ddb):
    S = 0
    for q, d in zip(Dq, Ddb):
        S = S + torch.from_numpy(q.astype(np.float32)).cuda() @ torch.from_numpy(d.astype(np.float32)).cuda().T
    return (S / len(Dq)).cpu().numpy()


def estimate(S, db_xyyaw, k=5, cluster_m=10.0, temp=0.02, mode='topk'):
    order = np.argsort(-S, 1)[:, :k]
    x, y, yaw = db_xyyaw
    best = order[:, 0]
    if mode == 'top1':
        return x[best], y[best], yaw[best], S[np.arange(len(S)), best]
    ex, ey, eyaw = np.zeros(len(S)), np.zeros(len(S)), np.zeros(len(S))
    for i in range(len(S)):
        o = order[i]
        near = np.hypot(x[o] - x[best[i]], y[o] - y[best[i]]) < cluster_m
        o = o[near]
        w = np.exp((S[i, o] - S[i, o].max()) / temp); w /= w.sum()
        ex[i], ey[i] = (w * x[o]).sum(), (w * y[o]).sum()
        eyaw[i] = np.angle((w * np.exp(1j * yaw[o])).sum())
    return ex, ey, eyaw, S[np.arange(len(S)), best]


def recall_at(err, ds=(2, 5, 10, 25)):
    return {f'recall@1_{d}m': float(np.mean(err < d) * 100) for d in ds}


def query_db(desc_dir, model, cams, qsplit='val', db_stride=1):
    """qsplit='val': queries = val laps, db = train. qsplit='tune': queries = the tune lap (inside train),
    db = train minus the tune lap +-10 s (for hyper-parameter selection without touching val)."""
    tdb, Ddb = aligned(desc_dir, model, cams, 'train')
    if qsplit == 'val':
        tq, Dq = aligned(desc_dir, model, cams, 'val')
    else:
        a, b = data.SPLITS[qsplit]; t0 = data.t0_ns()
        q = (tdb >= t0 + a * 1e9) & (tdb < t0 + b * 1e9)
        keep = (tdb < t0 + (a - 10) * 1e9) | (tdb >= t0 + (b + 10) * 1e9)
        tq, Dq = tdb[q], [d[q] for d in Ddb]
        tdb, Ddb = tdb[keep], [d[keep] for d in Ddb]
    return tq, Dq, tdb[::db_stride], [d[::db_stride] for d in Ddb]


def run(desc_dir, model, cams, out, k=5, cluster_m=10.0, temp=0.02, db_stride=1, mode='topk', name=None, video=False,
        qsplit='val'):
    os.makedirs(out, exist_ok=True)
    tq, Dq, tdb, Ddb = query_db(desc_dir, model, cams, qsplit, db_stride)
    db = data.interp_gt(tdb)
    S = similarity(Dq, Ddb)
    x, y, yaw, smax = estimate(S, db, k, cluster_m, temp, mode)
    pd.DataFrame(dict(timestamp_ns=tq, x=x, y=y, yaw=yaw)).to_csv(f'{out}/pred.csv', index=False, float_format='%.4f')
    gx, gy, _ = data.interp_gt(tq)
    err = np.hypot(x - gx, y - gy)
    ms = sum(float(np.load(f'{desc_dir}/{model}_{c}_train.npz')['ms_per_img']) for c in cams)
    extra = dict(method=f'{model} retrieval, cams={"+".join(cams)}, mode={mode}, k={k}, cluster={cluster_m} m, '
                        f'db_stride={db_stride} ({len(tdb)} db frames)',
                 runtime_ms_per_frame=ms, **recall_at(err))
    m = report.make_report(f'{out}/pred.csv', name or f'retrieval {model} {"+".join(cams)}', qsplit, video, extra)
    m.update(recall_at(err))
    np.savez(f'{out}/retrieval.npz', tq=tq, best=np.argmax(S, 1), top=np.argsort(-S, 1)[:, :50], smax=smax, err=err, tdb=tdb)
    json.dump(m, open(f'{out}/metrics_{qsplit}.json', 'w'), indent=1)
    return m


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--desc', required=True); ap.add_argument('--model', default='megaloc')
    ap.add_argument('--cams', nargs='+', default=['fl', 'fr', 'r']); ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=5); ap.add_argument('--cluster_m', type=float, default=10.0)
    ap.add_argument('--temp', type=float, default=0.02); ap.add_argument('--db_stride', type=int, default=1)
    ap.add_argument('--mode', default='topk'); ap.add_argument('--qsplit', default='val'); ap.add_argument('--name'); ap.add_argument('--video', action='store_true')
    a = ap.parse_args()
    m = run(a.desc, a.model, a.cams, a.out, a.k, a.cluster_m, a.temp, a.db_stride, a.mode, a.name, a.video, a.qsplit)
    print({k: round(v, 3) for k, v in m.items() if isinstance(v, float)})
