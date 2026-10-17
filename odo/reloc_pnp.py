"""Stage 2, iteration 2: metric refinement of the retrieval pose (method 3a of the brief, local version).

For each query (val image set, fl stamp) and each camera c:
  1. map frame A = training frame of camera c nearest to the retrieved database stamp, partner B = a later
     training frame with >= `min_base` m baseline. Their GT 6-DoF poses are the map (allowed: training GT).
  2. SuperPoint + LightGlue: A<->B -> triangulate 3D points in the map frame (reprojection / depth / angle checks);
     query<->A -> 2D(query)-3D correspondences through A's keypoints.
Then solve the base pose (x, y, yaw) of the query with z, roll, pitch taken from frame A's GT (same place on the
track), by robust least squares on the reprojection error over all cameras (Huber, then inlier refit),
initialised at the retrieval estimate. Falls back to the retrieval pose if too few inliers or a large jump.

    python -m odo.reloc_pnp --retrieval outputs/11_retrieval_megaloc --cams fr cl cr rr --qsplit val --out outputs/12_reloc_pnp
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from . import data, report
from .feature_vo import Extractor, load_gray
from .prep import CROP_BOTTOM


class GT6:
    def __init__(self):
        g = data.gt()
        self.t = g.timestamp_ns.values.astype(np.float64)
        self.p = g[['x', 'y', 'z']].values
        self.slerp = Slerp(self.t, Rotation.from_quat(g[['qx', 'qy', 'qz', 'qw']].values))

    def T(self, ts):
        ts = np.clip(np.atleast_1d(np.asarray(ts, dtype=np.float64)), self.t[0], self.t[-1])
        out = np.tile(np.eye(4), (len(ts), 1, 1))
        out[:, :3, :3] = self.slerp(ts).as_matrix()
        for k in range(3):
            out[:, k, 3] = np.interp(ts, self.t, self.p[:, k])
        return out


def undist(kp, K, D):
    return cv2.undistortPoints(kp.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)


def feats(ext, path, cam):
    im = load_gray(path, 1.0)
    f = ext.features(im)
    keep = f['keypoints'][0, :, 1] < CROP_BOTTOM[cam] * im.shape[0]
    return {k: v[:, keep] if v.dim() > 1 and v.shape[1] == keep.shape[0] else v for k, v in f.items()}


@torch.no_grad()
def match_idx(ext, f0, f1):
    m = ext.matcher({'image0': f0, 'image1': f1})
    return m['matches'][0].cpu().numpy()


def triangulate(nA, nB, TA, TB, max_reproj=2.0 / 1000, min_angle_deg=0.3, max_depth=200.0):
    """nA, nB normalized coords; TA, TB camera-to-map 4x4. Returns X (N,3) map frame and a validity mask."""
    PA = np.linalg.inv(TA)[:3]; PB = np.linalg.inv(TB)[:3]
    Xh = cv2.triangulatePoints(PA, PB, nA.T, nB.T)
    X = (Xh[:3] / Xh[3]).T
    ok = np.isfinite(X).all(1)
    def proj(P, X):
        c = X @ P[:3, :3].T + P[:3, 3]
        return c[:, :2] / c[:, 2:3], c[:, 2]
    pa, za = proj(PA, X); pb, zb = proj(PB, X)
    ok &= (za > 0.5) & (zb > 0.5) & (za < max_depth)
    ok &= (np.linalg.norm(pa - nA, axis=1) < max_reproj) & (np.linalg.norm(pb - nB, axis=1) < max_reproj)
    ra = X - TA[:3, 3]; rb = X - TB[:3, 3]
    cosang = (ra * rb).sum(1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1) + 1e-12)
    ok &= np.degrees(np.arccos(np.clip(cosang, -1, 1))) > min_angle_deg
    return X, ok


def base_pose(p, z, R_rp):
    x, y, yaw = p
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('z', yaw).as_matrix() @ R_rp
    T[:3, 3] = [x, y, z]
    return T


def residuals(p, z, R_rp, obs):
    Tb = base_pose(p, z, R_rp)
    res = []
    for Tbc, f, X, n in obs:
        Tcm = np.linalg.inv(Tb @ Tbc)
        c = X @ Tcm[:3, :3].T + Tcm[:3, 3]
        uv = c[:, :2] / np.maximum(c[:, 2:3], 1e-3)
        res.append(((uv - n) * f).ravel())
    return np.concatenate(res)


def refine_query(ext, gt6, tq, init, tdb_best, cams, min_base=1.5, max_jump=15.0, min_inl=30, dt_ns=0):
    obs, n_corr = [], 0
    TA_base = None
    for cam in cams:
        fr = data.frames(cam); st = fr.stamp_ns.values
        c = data.calib()[cam]; K, D, Tbc = c['K'], c['D'], c['T_base_cam']
        f_px = 0.5 * (K[0, 0] + K[1, 1])
        a = int(np.argmin(np.abs(st - tdb_best)))
        TbA = gt6.T(st[a] + dt_ns)[0]  # dt_ns: image-vs-GT clock offset (fitted on the tune lap)
        b = None
        for bb in list(range(a + 1, min(a + 8, len(st)))) + list(range(a - 1, max(a - 8, -1), -1)):
            if np.linalg.norm(gt6.T(st[bb] + dt_ns)[0][:3, 3] - TbA[:3, 3]) >= min_base:
                b = bb; break
        if b is None:
            continue
        TbB = gt6.T(st[b] + dt_ns)[0]
        q = int(np.argmin(np.abs(st - tq)))
        fA, fB, fQ = feats(ext, fr.path[a], cam), feats(ext, fr.path[b], cam), feats(ext, fr.path[q], cam)
        mAB = match_idx(ext, fA, fB); mQA = match_idx(ext, fQ, fA)
        kA = fA['keypoints'][0].cpu().numpy(); kB = fB['keypoints'][0].cpu().numpy(); kQ = fQ['keypoints'][0].cpu().numpy()
        X, ok = triangulate(undist(kA[mAB[:, 0]], K, D), undist(kB[mAB[:, 1]], K, D), TbA @ Tbc, TbB @ Tbc,
                            max_reproj=2.0 / f_px)
        X3 = {int(i): X[j] for j, i in enumerate(mAB[:, 0]) if ok[j]}
        sel = [(int(iq), X3[int(ia)]) for iq, ia in mQA if int(ia) in X3]
        if len(sel) < 6:
            continue
        nQ = undist(kQ[[s[0] for s in sel]], K, D)
        obs.append((Tbc, f_px, np.array([s[1] for s in sel]), nQ))
        n_corr += len(sel)
        if TA_base is None:
            TA_base = TbA
    if not obs or n_corr < min_inl:
        return init, dict(status='few_corr', n_corr=n_corr, n_inl=0)
    z = TA_base[2, 3]
    yawA = np.arctan2(TA_base[1, 0], TA_base[0, 0])
    R_rp = Rotation.from_euler('z', -yawA).as_matrix() @ TA_base[:3, :3]
    p0 = np.array(init, dtype=np.float64)
    r1 = least_squares(residuals, p0, args=(z, R_rp, obs), loss='huber', f_scale=3.0, x_scale=[1, 1, 0.01])
    # inlier refit
    res = residuals(r1.x, z, R_rp, obs).reshape(-1, 2)
    inl = np.linalg.norm(res, axis=1) < 3.0
    obs2, k = [], 0
    for Tbc, f, X, n in obs:
        m = inl[k:k + len(X)]; k += len(X)
        if m.sum() >= 3:
            obs2.append((Tbc, f, X[m], n[m]))
    n_inl = int(inl.sum())
    if n_inl < min_inl:
        return init, dict(status='few_inl', n_corr=n_corr, n_inl=n_inl)
    r2 = least_squares(residuals, r1.x, args=(z, R_rp, obs2), loss='soft_l1', f_scale=1.0, x_scale=[1, 1, 0.01])
    if np.hypot(r2.x[0] - p0[0], r2.x[1] - p0[1]) > max_jump:
        return init, dict(status='jump', n_corr=n_corr, n_inl=n_inl)
    rms = float(np.sqrt(np.mean(residuals(r2.x, z, R_rp, obs2) ** 2)))
    x, y, yaw = r2.x
    if dt_ns:
        st_ = data.interp_state([tq], ('v_mps', 'wz_radps'))
        d = st_['v_mps'][0] * dt_ns / 1e9
        x, y, yaw = x - d * np.cos(yaw), y - d * np.sin(yaw), yaw - st_['wz_radps'][0] * dt_ns / 1e9
    return (x, y, yaw), dict(status='ok', n_corr=n_corr, n_inl=n_inl, rms_px=rms, n_cams=len(obs2))


def main(retr_dir, cams, out, qsplit='val', name=None, video=False, limit=None, n_cand=3, fallback_cams=None, **kw):
    os.makedirs(out, exist_ok=True)
    rz = np.load(f'{retr_dir}/retrieval.npz')
    pred = pd.read_csv(f'{retr_dir}/pred.csv')
    tq, best, tdb = rz['tq'], rz['best'], rz['tdb']
    ext = Extractor('superpoint', 2048)
    gt6 = GT6()
    rows, infos = [], []
    t0 = time.time()
    n = len(tq) if limit is None else min(limit, len(tq))
    top = rz['top'] if 'top' in rz.files else best[:, None]
    dbxy = np.stack(data.interp_gt(tdb)[:2], 1)
    for i in range(n):
        init = (pred.x[i], pred.y[i], pred.yaw[i])
        # geometric re-ranking: try up to n_cand spatially distinct retrieval candidates (>= 8 m apart)
        cands = []
        for j in top[i]:
            if all(np.linalg.norm(dbxy[j] - dbxy[c]) > 8.0 for c in cands):
                cands.append(j)
            if len(cands) == n_cand:
                break
        for ci, j in enumerate(cands):
            init_c = init if ci == 0 else (dbxy[j][0], dbxy[j][1], data.interp_gt([tdb[j]])[2][0])
            p, info = refine_query(ext, gt6, tq[i], init_c, tdb[j], cams, **kw)
            info['cand'] = ci
            if info['status'] == 'ok':
                break
        if info['status'] != 'ok' and fallback_cams:
            # hard case: retry the candidates with every camera
            for ci, j in enumerate(cands):
                init_c = init if ci == 0 else (dbxy[j][0], dbxy[j][1], data.interp_gt([tdb[j]])[2][0])
                p, info = refine_query(ext, gt6, tq[i], init_c, tdb[j], fallback_cams, **kw)
                info['cand'] = 10 + ci
                if info['status'] == 'ok':
                    break
        if info['status'] != 'ok':
            p = init
        rows.append((tq[i], p[0], p[1], np.angle(np.exp(1j * p[2]))))
        infos.append(info)
        if i % 200 == 0:
            print(i, n, f'{(time.time() - t0) / (i + 1) * 1000:.0f} ms/query', info, flush=True)
    ms = (time.time() - t0) / n * 1000
    pd.DataFrame(rows, columns=['timestamp_ns', 'x', 'y', 'yaw']).to_csv(f'{out}/pred.csv', index=False, float_format='%.4f')
    inf = pd.DataFrame(infos); inf.to_csv(f'{out}/refine_info.csv', index=False)
    stat = inf.status.value_counts(normalize=True).mul(100).round(1).to_dict()
    extra = dict(method=f'retrieval ({os.path.basename(retr_dir)}) + triangulated map frame + 3-DoF robust PnP',
                 cams=' '.join(cams), dt_ms=kw.get('dt_ns', 0) / 1e6, runtime_ms_per_frame=ms, **{f'status_{k}_pct': v for k, v in stat.items()},
                 median_inliers=float(inf.n_inl.median()), n_cand=n_cand,
                 cand_used_pct=json.dumps(inf.cand.value_counts(normalize=True).mul(100).round(1).to_dict()))
    m = report.make_report(f'{out}/pred.csv', name or 'retrieval + PnP', qsplit, video, extra,
                           baselines={'retrieval only': f'{retr_dir}/metrics_{qsplit}.json'},
                           others={'retrieval only': f'{retr_dir}/pred.csv'})
    print({k: round(v, 3) for k, v in m.items() if isinstance(v, float)})
    return m


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--retrieval', required=True); ap.add_argument('--cams', nargs='+', default=['fr', 'cl', 'cr', 'rr'])
    ap.add_argument('--out', required=True); ap.add_argument('--qsplit', default='val'); ap.add_argument('--name')
    ap.add_argument('--video', action='store_true'); ap.add_argument('--limit', type=int)
    ap.add_argument('--fallback_cams', nargs='*', default=data.CAMS); ap.add_argument('--n_cand', type=int, default=3); ap.add_argument('--min_base', type=float, default=1.5); ap.add_argument('--dt_ms', type=float, default=0.0)
    a = ap.parse_args()
    main(a.retrieval, a.cams, a.out, a.qsplit, a.name, a.video, a.limit, a.n_cand, a.fallback_cams, min_base=a.min_base, dt_ns=int(a.dt_ms * 1e6))
