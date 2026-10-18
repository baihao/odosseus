"""Probe: measurement error per map pass at given query times (prior = GT pose at the GT-clock time).
Shows whether some training passes give inconsistent (biased) fixes.
    python scripts/probe_passes.py tune fr,cl 712 718 0.5 [fold]"""
import sys; sys.path.insert(0, '.')
import numpy as np
from odo import data, online
split, cams = sys.argv[1], sys.argv[2].split(',')
ta, tb, step = map(float, sys.argv[3:6])
t0 = data.t0_ns(); a, b = data.SPLITS[split]
win = data.fold_windows(sys.argv[6]) if len(sys.argv) > 6 else [(0, t0 + int(897e9))] if split == 'val' else [(0, t0 + int((a - 10) * 1e9)), (t0 + int((b + 10) * 1e9), t0 + int(897e9))]
cfg = dict(online.DEFAULT)
cams_all = cams
fmap = online.FeatureMap('outputs/20_map', cams, win, cfg['max_gt_acc']); cm = online.Cams(cams); fe = online.Frontend(cams)
sets = online.image_sets(split, cams)
st = sets.stamp_ns.values
for tq in np.arange(ta, tb, step):
    k = int(np.argmin(np.abs(st - (t0 + tq * 1e9))))
    tm = st[k] - 9_100_000
    gx, gy, gyaw = [v[0] for v in data.interp_gt([tm])]
    feats = fe.features(fe.load(list(sets.iloc[k][cams])))
    sel = {c: fmap.select(c, gx, gy, gyaw, 8.0, 8) for c in cams}
    out = []
    for j in range(len(sel[cams[0]])):
        frames = {c: sel[c][j:j + 1] if j < len(sel[c]) else sel[c][:0] for c in cams}
        p, cov, info = online.measure(fmap, cm, feats, np.array([gx, gy, gyaw]), cfg, 24.0, frames)
        ts_map = fmap.cams[cams[0]]['stamp'][sel[cams[0]][j]]
        if p is None:
            out.append(f'{(ts_map - t0) / 1e9:.0f}s:none'); continue
        dx, dy = p[0] - gx, p[1] - gy
        lat = -np.sin(gyaw) * dx + np.cos(gyaw) * dy; lon = np.cos(gyaw) * dx + np.sin(gyaw) * dy
        out.append(f'{(ts_map - t0) / 1e9:.0f}s:lon{lon:+.2f}/lat{lat:+.2f}/n{info["n_inl"]}')
    p, cov, info = online.measure(fmap, cm, feats, np.array([gx, gy, gyaw]), cfg, 24.0)
    e = np.hypot(p[0] - gx, p[1] - gy) if p is not None else np.nan
    print(f't={tq:.1f} all3: {e:.2f} m n{info["n_inl"]} {info.get("inl_cam")} |', ' '.join(out), flush=True)
