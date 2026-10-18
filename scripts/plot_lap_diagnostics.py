"""Diagnostic figure for one lap of one CV fold (shared time axis):
  1 position error, 2 lateral error, 3 distance of the driven line (GT) to the nearest map-lap pose ("off the mapped
  lines"), 4 guided-matching quality (stage-1 matches, PnP inliers), 5 inliers per camera group, 6 MegaLoc retrieval
  against the fold's map (top-1 cosine similarity), 7 distance of the MegaLoc top-1 map frame to the GT, 8 speed,
  9 yaw rate, 10 GT position accuracy (state estimator's own flag). Grey strip in row 1: frames without an accepted fix.
    python scripts/plot_lap_diagnostics.py F5 L2 -> outputs/21_online/cv/F5/L2/diagnostics.{png,pdf}, docs/figures/F5_L2_diagnostics.png"""
import sys, shutil; sys.path.insert(0, '.')
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from odo import data, report
from odo.retrieval import aligned
fold, lap = sys.argv[1], sys.argv[2]
d = f'outputs/21_online/cv/{fold}/{lap}'
m, per = report.compute_all(report.load_pred(f'{d}/pred.csv'), lap)
ts = per['ts']; t = (ts - data.t0_ns()) / 1e9
inf = pd.read_csv(f'{d}/online_info.csv').set_index('timestamp_ns').reindex(ts)
t0 = data.t0_ns()
inmap = lambda tt: np.any([(tt >= t0 + data.LAPS[l][0] * 1e9) & (tt < t0 + data.LAPS[l][1] * 1e9) for l in data.FOLDS[fold]['map_laps']], 0)

# distance of the driven line to the nearest map-lap pose (GT of the map laps)
g = data.gt(); gm = g[inmap(g.timestamp_ns.values)]
dist_map, _ = cKDTree(gm[['x', 'y']].values).query(np.stack([per['gx'], per['gy']], 1))

# MegaLoc retrieval of every query frame against the fold's map frames (7 cameras, averaged cosine similarity)
cams = data.CAMS; Tq, Dq, Tdb, Ddb = [], [], [], []
for split in ('train', 'val', 'L8'):
    tt, dd = aligned('outputs/10_vpr_desc', 'megaloc', cams, split, ref='fl')
    dd = np.stack(dd, 0)
    sel_db = inmap(tt); Tdb.append(tt[sel_db]); Ddb.append(dd[:, sel_db])
    Tq.append(tt); Dq.append(dd)
Tq = np.concatenate(Tq); Dq = np.concatenate(Dq, 1); Tdb = np.concatenate(Tdb); Ddb = np.concatenate(Ddb, 1)
j = np.clip(np.searchsorted(Tq, ts), 1, len(Tq) - 1); j = np.where(np.abs(Tq[j - 1] - ts) < np.abs(Tq[j] - ts), j - 1, j)
Q = torch.from_numpy(Dq[:, j]).cuda().float(); DB = torch.from_numpy(Ddb).cuda().float()
S = sum(Q[c] @ DB[c].T for c in range(len(cams))) / len(cams)
top_s, top_i = S.max(1)
top_s = top_s.cpu().numpy(); top_i = top_i.cpu().numpy()
dbx, dby, _ = data.interp_gt(Tdb[top_i])
dist_top1 = np.hypot(dbx - per['gx'], dby - per['gy'])

st = data.interp_state(ts, ('v_mps', 'wz_radps'))
sd = data.state(); acc = np.hypot(sd['pos_accuracy.0'], sd['pos_accuracy.1']).values
gacc = np.interp(ts.astype(float), sd.timestamp_ns.values.astype(float), acc)
nofix = (inf.status != 'fix').values
ic = inf.inl_cam.fillna('0/0/0/0/0/0/0').str.split('/', expand=True).astype(int).values  # fl fr cl cr r rl rr
groups = [('front (fl, fr)', [0, 1]), ('side front (cl, cr)', [2, 3]), ('rear (r, rl, rr)', [4, 5, 6])]
C = report.SERIES

fig, axs = plt.subplots(10, 1, figsize=(13, 17), sharex=True)
a = axs[0]; a.plot(t, per['pos'], color=C[0], lw=1); a.axhline(1, color=report.MUTED, ls='--', lw=.8); a.axhline(.25, color=report.MUTED, ls='--', lw=.8)
a.fill_between(t, -0.05 * per['pos'].max(), 0, where=nofix, color=report.MUTED, step='mid', lw=0)
a.set_ylabel('position\nerror [m]')
a = axs[1]; a.plot(t, per['lat'], color=C[0], lw=1); a.axhline(0, color=report.MUTED, lw=.8); a.set_ylabel('lateral\nerror [m]')
a = axs[2]; a.plot(t, dist_map, color=C[0], lw=1); a.set_ylabel('distance to\nnearest map\nlap pose [m]')
a = axs[3]; a.plot(t, inf.n_m1, color=C[0], lw=1, label='guided matches (stage 1)'); a.plot(t, inf.n_inl, color=C[1], lw=1, label='PnP inliers')
a.set_yscale('symlog', linthresh=10); a.set_ylabel('matches'); a.legend(loc='lower right', bbox_to_anchor=(1, 1.0), frameon=False, fontsize=8, ncol=2, borderaxespad=0.1)
a = axs[4]
for k, (lab, idx) in enumerate(groups):
    a.plot(t, ic[:, idx].sum(1), color=C[k], lw=1, label=lab)
a.set_yscale('symlog', linthresh=10); a.set_ylabel('PnP inliers\nper camera\ngroup'); a.legend(loc='lower right', bbox_to_anchor=(1, 1.0), frameon=False, fontsize=8, ncol=3, borderaxespad=0.1)
a = axs[5]; a.plot(t, top_s, color=C[0], lw=1); a.set_ylabel('MegaLoc top-1\ncosine sim.\n(vs map)')
a = axs[6]; a.plot(t, np.maximum(dist_top1, 0.1), color=C[0], lw=1); a.set_yscale('log'); a.axhline(10, color=report.MUTED, ls='--', lw=.8)
a.set_ylabel('MegaLoc top-1\nframe distance\nto GT [m]')
a = axs[7]; a.plot(t, st['v_mps'], color=C[0], lw=1); a.set_ylabel('speed [m/s]')
a = axs[8]; a.plot(t, np.degrees(st['wz_radps']), color=C[0], lw=1); a.axhline(0, color=report.MUTED, lw=.8); a.set_ylabel('yaw rate [°/s]')
a = axs[9]; a.plot(t, gacc, color=C[0], lw=1); a.set_yscale('log'); a.set_ylabel('GT pos.\naccuracy flag [m]'); a.set_xlabel('run time t [s]')
for a in axs:
    a.margins(x=0)
# mark the worst moments
for pk in np.argsort(-per['pos'])[:1]:
    for a in axs:
        a.axvline(t[pk], color=report.INK2, lw=0.6, ls=':')
fig.suptitle(f'Exp21 {fold} / {lap}: error and its context   (RMSE {m["pos_rmse"]:.3f} m, max {per["pos"].max():.2f} m at t = {t[np.argmax(per["pos"])]:.1f} s)',
             x=0.01, ha='left', fontsize=12, weight='bold', color=report.INK)
fig.text(0.01, 0.003, f'Map laps {" ".join(data.FOLDS[fold]["map_laps"])}. Row 1: grey strip = no accepted fix; dashed 0.25 m / 1 m. '
         'Row 7: dashed = 10 m (retrieval "correct" threshold).\nDotted vertical line: largest error. MegaLoc retrieval '
         'computed offline against the map laps (the online system uses it only when lost).', fontsize=8, color=report.INK2)
fig.tight_layout(rect=(0, 0.02, 1, 0.98))
out = f'{d}/diagnostics'
fig.savefig(out + '.png', dpi=100); fig.savefig(out + '.pdf')
shutil.copy(out + '.png', f'docs/figures/{fold}_{lap}_diagnostics.png')
pk = np.argmax(per['pos']); w = np.abs(t - t[pk]) < 3
print(f'saved {out}; worst t={t[pk]:.1f}: err {per["pos"][pk]:.2f}, dist to map {dist_map[pk]:.1f} m, inliers {inf.n_inl.values[pk]}, '
      f'megaloc top1 sim {top_s[pk]:.3f} dist {dist_top1[pk]:.1f} m; lap medians: dist_map {np.median(dist_map):.1f}, top1 sim {np.median(top_s):.3f}, '
      f'top1 within 10 m {np.mean(dist_top1 < 10) * 100:.1f} %; corr(err, dist_map) {np.corrcoef(per["pos"], dist_map)[0, 1]:.2f}')
