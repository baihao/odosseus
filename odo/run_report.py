"""Per-run error report of the online localiser: <out>/run_report.pdf.

Reads <out>/pred.csv and <out>/online_info.csv (odo.online) and the GT of the run's lap/segment.
  page 1  summary: score metrics, fix statistics (map fixes, gated fixes, global-retrieval attempts / accepted)
  page 2  over time: position, lateral and longitudinal error; innovation (distance between each fix and the
          dead-reckoning prediction, the "relative tracking"), lateral/longitudinal; fix rate (share of frames with an
          accepted map fix, gated fixes, global-retrieval attempts and accepted retrieval fixes, 2 s windows);
          PnP inliers
  page 3  the same, zoomed to +-15 s around the largest position error
and <out>/summary.md: setup (segment, map, config changes vs odo.online.DEFAULT), metrics vs the reference run, fix and
retrieval statistics, and the worst error episodes (> 0.5 m) with what the localiser saw during each of them.
An optional reference run (e.g. the same lap on night images) is drawn in grey in the error panels.

    python -m odo.run_report <out_dir> --name "<title>" [--ref <other out_dir> --ref_name night]
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import data, metrics
from .report import INK, INK2, MUTED, SERIES

FIXED = ('fix', 'reloc', 'reset')            # statuses whose fix entered the filter


def load(out):
    p = pd.read_csv(f'{out}/pred.csv').sort_values('timestamp_ns').reset_index(drop=True)
    inf = pd.read_csv(f'{out}/online_info.csv')
    ts = p.timestamp_ns.values
    gx, gy, gyaw = data.interp_gt(ts)
    dx, dy = p.x.values - gx, p.y.values - gy
    c, s = np.cos(gyaw), np.sin(gyaw)
    r = pd.DataFrame(dict(t=(ts - data.t0_ns()) / 1e9, pos=np.hypot(dx, dy), lon=c * dx + s * dy, lat=-s * dx + c * dy))
    for k in ('status', 'n_inl', 'innov', 'innov_lat', 'innov_lon', 'reloc_try', 'map_dist'):
        r[k] = inf[k].values if k in inf else np.nan
    r['status'] = r.status.fillna('none')
    r['reloc_try'] = r.reloc_try.fillna(0)
    return r


def rate(t, flag, win=2.0):
    """Share of frames [%] with flag, centred window of win seconds."""
    f = pd.Series(flag.astype(float).values, index=pd.to_datetime(t, unit='s'))
    return f.rolling(f'{win}s', center=True).mean().values * 100


def summary(r):
    st = r.status.value_counts(normalize=True).mul(100)
    acc = r.status.isin(FIXED)
    return {
        'frames': len(r), 'position RMSE [m]': np.sqrt(np.mean(r.pos ** 2)), 'position median [m]': r.pos.median(),
        'position p95 [m]': np.percentile(r.pos, 95), 'max [m]': r.pos.max(),
        'lateral RMSE [m]': np.sqrt(np.mean(r.lat ** 2)), 'longitudinal RMSE [m]': np.sqrt(np.mean(r.lon ** 2)),
        'within 1 m [%]': np.mean(r.pos <= 1) * 100,
        'map fix accepted [%]': np.mean(r.status == 'fix') * 100, 'gated (rejected) fix [%]': st.get('gated', 0),
        'no fix [%]': np.mean(~r.status.isin(['fix', 'gated', 'reloc', 'reloc_rejected', 'reset'])) * 100,
        'challenger reset [%]': st.get('reset', 0),
        'global retrieval attempts [% frames]': np.mean(r.reloc_try > 0) * 100,
        'global retrieval fix accepted [% frames]': st.get('reloc', 0),
        'global retrieval fix rejected [% frames]': st.get('reloc_rejected', 0),
        'innovation median / p95 [m]': (np.nanmedian(r.innov[acc]), np.nanpercentile(r.innov[acc], 95))
        if r.innov[acc].notna().any() else (np.nan, np.nan),
        'PnP inliers median': r.n_inl.median(),
    }


def _panels(axs, r, ref, ref_name, name):
    t = r.t.values
    for ax, k, lab in ((axs[0], 'pos', 'position err [m]'), (axs[1], 'lat', 'lateral err [m]'),
                       (axs[2], 'lon', 'longit. err [m]')):
        if ref is not None:
            ax.plot(ref.t, ref[k], color=MUTED, lw=1.0, label=ref_name)
        ax.plot(t, r[k], color=SERIES[0], lw=1.2, label=name)
        ax.set_ylabel(lab)
        if k != 'pos':
            ax.axhline(0, color=INK2, lw=0.6)
    if ref is not None:
        axs[0].legend(loc='upper left', frameon=False, fontsize=8)
    # innovation: fix minus dead-reckoning prediction (car frame); accepted fixes vs gated ones
    acc = r.status.isin(FIXED).values; gat = r.status.isin(['gated', 'reloc_rejected']).values
    ax = axs[3]
    ax.plot(t[acc], r.innov_lat[acc], '.', ms=2.5, color=SERIES[0], label='lateral (accepted fix)')
    ax.plot(t[acc], r.innov_lon[acc], '.', ms=2.5, color=SERIES[1], label='longitudinal (accepted fix)')
    ax.plot(t[gat], r.innov[gat], 'x', ms=4, color=INK, label='|innovation| of gated fix')
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylabel('fix − DR pred. [m]')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False, fontsize=7, ncol=3, borderaxespad=0.1)
    # fix rate
    ax = axs[4]
    ax.plot(t, rate(t, r.status == 'fix'), color=SERIES[0], label='map fix accepted')
    ax.plot(t, rate(t, r.status.isin(['gated', 'reloc_rejected'])), color=SERIES[1], label='fix gated')
    ax.plot(t, rate(t, r.reloc_try > 0), color=SERIES[2], label='global retrieval attempt')
    ax.plot(t, rate(t, r.status.isin(['reloc', 'reset'])), color=INK, lw=1.0, label='retrieval / reset fix accepted')
    ax.set_ylim(-3, 103); ax.set_ylabel('fix rate [% frames, 2 s]')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False, fontsize=7, ncol=4, borderaxespad=0.1)
    ax = axs[5]
    if ref is not None:
        ax.plot(ref.t, ref.n_inl.clip(lower=1), color=MUTED, lw=1.0)
    ax.plot(t, r.n_inl.clip(lower=1), color=SERIES[0], lw=1.0)
    ax.set_yscale('log'); ax.set_ylabel('PnP inliers'); ax.set_xlabel('t since run start [s]')


def episodes(r, thr=0.5, merge_s=2.0):
    """Contiguous stretches with position error > thr (gaps < merge_s merged), worst first."""
    hi = r.pos.values > thr
    t = r.t.values
    eps, start = [], None
    for k in range(len(r)):
        if hi[k] and start is None:
            start = k
        if start is not None and (not hi[k] or k == len(r) - 1):
            end = k if hi[k] else k - 1
            if eps and t[start] - t[eps[-1][1]] < merge_s:
                eps[-1] = (eps[-1][0], end)
            else:
                eps.append((start, end))
            start = None
    return sorted(eps, key=lambda e: -r.pos.values[e[0]:e[1] + 1].max())


def _fmt(v):
    if isinstance(v, tuple):
        return ' / '.join(_fmt(x) for x in v)
    if isinstance(v, (int, np.integer)):
        return f'{v}'
    return f'{v:.3f}' if abs(v) < 10 else f'{v:.1f}'


def write_summary(out, name, r, sm, rr=None, sr=None, ref_name='reference', meta=None, notes=None):
    from .online import DEFAULT
    cfgf = f'{out}/config.json'
    cj = json.load(open(cfgf)) if os.path.exists(cfgf) else {}
    # options added after a run was made were off in it (Exp21 values)
    cfg = dict(dict(reloc_yaw0_deg=1.0, reloc_yaw_rate_deg=2.0, yaw_min_inl=0, sim1_lo=0.0, mask_bright=0, kp_keep=1.0),
               **cj.get('cfg', {}))
    diff = {k: v for k, v in cfg.items() if k not in DEFAULT or DEFAULT[k] != v}
    L = [f'# {name}', '', f'Output folder: `{out}`. Files: `pred.csv` (submission format), `online_info.csv` (per-frame '
         'localiser log), `run_report.pdf` (error, innovation, fix rate over time; zoom on the worst error), '
         '`report_<lap>.pdf` / `traj_<lap>.html` (map and metric pages), `metrics_<lap>.json`, `config.json`.', '',
         '## Setup', '',
         f'* segment: `{cj.get("split", "?")}`, t = {r.t.iloc[0]:.1f} .. {r.t.iloc[-1]:.1f} s since the run start, '
         f'{len(r)} image sets, cameras {" ".join(cj.get("cams", []))}']
    for k, v in (meta or {}).items():
        L.append(f'* {k}: {v}')
    L.append('* configuration changes vs `odo.online.DEFAULT`: ' +
             (', '.join(f'`{k}={json.dumps(v)}`' for k, v in diff.items()) if diff else 'none (default system)'))
    L += ['', '## Result', '', f'| | this run |' + (f' {ref_name} |' if sr else ''), '|---|---|' + ('---|' if sr else '')]
    for k, v in sm.items():
        f = (lambda x: f'{x:.1f}') if '%' in k else _fmt
        L.append(f'| {k} | {f(v)} |' + (f' {f(sr[k])} |' if sr else ''))
    L += ['', 'Definitions: *map fix accepted* = guided matching to the night map + PnP that passed the filter gate; '
          '*gated* = fix rejected by the Mahalanobis gate; *global retrieval* = MegaLoc retrieval + wide-radius PnP, tried '
          'after 1 s without an accepted fix; *innovation* = fix minus the dead-reckoning prediction.', '']
    eps = episodes(r)
    L += ['## Worst error episodes (position error > 0.5 m)', '']
    if not eps:
        L.append(f'None: the error stays below 0.5 m (max {r.pos.max():.2f} m at t = {r.t[r.pos.idxmax()]:.1f} s).')
    else:
        L += [f'{len(eps)} episode(s), {sum(e[1] - e[0] + 1 for e in eps) / len(r) * 100:.1f} % of the frames.', '',
              '| t start–end [s] | dur [s] | peak [m] | lat / lon at peak [m] | map fix / gated / no fix [%] | '
              'retrieval tries (accepted) | median inliers | median dist. to map lines [m] |',
              '|---|---|---|---|---|---|---|---|']
        for a, b in eps[:8]:
            e = r.iloc[a:b + 1]; k = e.pos.idxmax()
            st = e.status
            L.append(f'| {e.t.iloc[0]:.1f}–{e.t.iloc[-1]:.1f} | {e.t.iloc[-1] - e.t.iloc[0] + 0.1:.1f} | {e.pos.max():.2f} | '
                     f'{r.lat[k]:+.2f} / {r.lon[k]:+.2f} | {np.mean(st == "fix") * 100:.0f} / '
                     f'{np.mean(st.isin(["gated", "reloc_rejected"])) * 100:.0f} / {np.mean(st == "none") * 100:.0f} | '
                     f'{int((e.reloc_try > 0).sum())} ({int(st.isin(["reloc", "reset"]).sum())}) | '
                     f'{e.n_inl.median():.0f} | {e.map_dist.median():.1f} |')
    if notes:
        L += ['', '## Interpretation', '', notes]
    open(f'{out}/summary.md', 'w').write('\n'.join(L) + '\n')


def make(out, name, ref=None, ref_name='reference', pdf_name='run_report.pdf', meta=None, notes=None):
    r = load(out)
    rr = load(ref) if ref else None
    sm = summary(r)
    write_summary(out, name, r, sm, rr, summary(rr) if rr is not None else None, ref_name, meta, notes)
    with PdfPages(f'{out}/{pdf_name}') as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, name, fontsize=14, color=INK, weight='bold')
        fig.text(0.07, 0.925, f'{out}   t = {r.t.iloc[0]:.0f} .. {r.t.iloc[-1]:.0f} s', color=INK2, fontsize=8)
        rows = []
        sr = summary(rr) if rr is not None else None
        fmt = lambda v: (f'{v[0]:.3f} / {v[1]:.3f}' if isinstance(v, tuple) else
                         (f'{v:.0f}' if isinstance(v, (int, np.integer)) else f'{v:.3f}' if abs(v) < 10 else f'{v:.1f}'))
        for k, v in sm.items():
            rows.append([k, fmt(v)] + ([fmt(sr[k])] if sr is not None else []))
        ax = fig.add_axes([0.07, 0.3, 0.86, 0.6]); ax.axis('off')
        tb = ax.table(cellText=rows, colLabels=['', 'this run'] + ([ref_name] if sr is not None else []),
                      loc='upper left', cellLoc='left', colLoc='left',
                      colWidths=[0.5, 0.25, 0.25] if sr is not None else [0.6, 0.3])
        tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.5)
        for (i, j), cell in tb.get_celld().items():
            cell.set_edgecolor(MUTED)
            if i == 0:
                cell.set_text_props(weight='bold')
        fig.text(0.07, 0.27, 'map fix = guided matching to the night map + PnP accepted by the filter gate; gated = fix '
                 'rejected by the\nMahalanobis gate; global retrieval = MegaLoc retrieval + wide-radius PnP, tried after '
                 '1 s without a fix;\ninnovation = fix minus the dead-reckoning (car-state) prediction, in the car frame.',
                 fontsize=8, color=INK2, va='top')
        pdf.savefig(fig); plt.close(fig)
        for zoom in (False, True):
            fig, axs = plt.subplots(6, 1, figsize=(11.69, 13), sharex=True,
                                    gridspec_kw=dict(height_ratios=[1, 1, 1, 1.2, 1.2, 0.8]))
            tm = r.t[r.pos.idxmax()]
            cut = (lambda d: d[(d.t > tm - 15) & (d.t < tm + 15)]) if zoom else (lambda d: d)
            _panels(axs, cut(r), cut(rr) if rr is not None else None, ref_name, 'this run')
            if zoom:
                fig.suptitle(f'{name}: ±15 s around the largest error ({r.pos.max():.2f} m at t = {tm:.1f} s)', color=INK)
            else:
                fig.suptitle(f'{name}: error, innovation and fix rate over time', color=INK)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return sm


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--name', default=None)
    ap.add_argument('--ref'); ap.add_argument('--ref_name', default='reference'); ap.add_argument('--notes')
    a = ap.parse_args()
    s = make(a.out, a.name or a.out, a.ref, a.ref_name, notes=a.notes)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
