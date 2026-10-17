"""Experiment reports: metrics.json, report.pdf, interactive traj.html, optional debug video.

    python -m odo.report outputs/<exp>/pred.csv --name "<title>" [--split val] [--video]

pred.csv: timestamp_ns,x,y,yaw (same format as the challenge submission).
"""
import argparse
import json
import os

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection

from . import data, metrics

# reference palette (dataviz skill): neutral ink for GT, categorical slots for methods
INK = '#0b0b0b'; INK2 = '#52514e'; MUTED = '#b9b8b2'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
                     'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'lines.linewidth': 1.5})

METRIC_ROWS = [
    ('pos_rmse', 'position RMSE [m]  (SCORE)', '{:.3f}'),
    ('pos_median', 'position median [m]', '{:.3f}'),
    ('pos_p95', 'position p95 [m]', '{:.3f}'),
    ('lat_rmse', 'lateral RMSE [m]', '{:.3f}'),
    ('lon_rmse', 'longitudinal RMSE [m]', '{:.3f}'),
    ('yaw_rmse_deg', 'yaw RMSE [deg]', '{:.3f}'),
    ('within_025', 'within 0.25 m [%]', '{:.1f}'),
    ('within_050', 'within 0.50 m [%]', '{:.1f}'),
    ('within_100', 'within 1.00 m [%]', '{:.1f}'),
    ('coverage_pct', 'coverage (1 s bins) [%]', '{:.1f}'),
    ('rpe_1s_trans_rmse_m', 'RPE 1 s translation RMSE [m]', '{:.3f}'),
    ('rpe_1s_yaw_rmse_deg', 'RPE 1 s yaw RMSE [deg]', '{:.3f}'),
    ('rpe_10s_trans_pct', 'RPE 10 s translation drift [%]', '{:.2f}'),
    ('rpe_10s_yaw_rmse_deg', 'RPE 10 s yaw RMSE [deg]', '{:.3f}'),
    ('ate_se2', 'ATE after SE(2) align [m] (diag.)', '{:.3f}'),
    ('runtime_ms_per_frame', 'runtime [ms/frame]', '{:.1f}'),
]


def load_pred(path):
    p = pd.read_csv(path)
    return p.sort_values('timestamp_ns').reset_index(drop=True)


def compute_all(pred, split):
    a, b = data.SPLITS[split]
    t0 = data.t0_ns()
    tg_last = int(data.gt().timestamp_ns.iloc[-1])
    rng = (int(t0 + a * 1e9), int(min(t0 + b * 1e9, tg_last)))
    sel = (pred.timestamp_ns >= rng[0]) & (pred.timestamp_ns < rng[1])
    p = pred[sel]
    m, per = metrics.challenge_metrics(p.timestamp_ns.values, p.x.values, p.y.values, p.yaw.values, rng)
    m.update(metrics.relative_errors(p.timestamp_ns.values, p.x.values, p.y.values, p.yaw.values))
    m['ate_se2'], _ = metrics.ate(p.timestamp_ns.values, p.x.values, p.y.values, p.yaw.values)
    return m, per


def _track(ax):
    for k, v in data.track_bounds().items():
        ax.plot(v[:, 0], v[:, 1], color=MUTED, lw=0.6, zorder=0)


def _colored(ax, x, y, c, vmax, cmap='Blues'):
    pts = np.stack([x, y], 1).reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], 1), cmap=cmap, norm=plt.Normalize(0, vmax))
    lc.set_array(c[:-1]); lc.set_linewidth(2.0)
    ax.add_collection(lc)
    return lc


def pdf_report(out_pdf, name, m, per, extra_text='', baselines=None):
    t = (per['ts'] - data.t0_ns()) / 1e9
    with PdfPages(out_pdf) as pdf:
        # page 1: metric table
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, name, fontsize=15, color=INK, weight='bold')
        fig.text(0.07, 0.925, f'{m.get("split", "")}: {m["n_evaluated"]} poses, '
                              f't = {t[0]:.0f} .. {t[-1]:.0f} s', color=INK2)
        cols = ['metric', 'this run'] + [b for b in (baselines or {})]
        rows = []
        for k, lab, f in METRIC_ROWS:
            if k not in m and not any(k in bm for bm in (baselines or {}).values()):
                continue
            r = [lab, f.format(m[k]) if k in m and m[k] is not None else '-']
            for bm in (baselines or {}).values():
                r.append(f.format(bm[k]) if k in bm and bm[k] is not None else '-')
            rows.append(r)
        ax = fig.add_axes([0.07, 0.45, 0.86, 0.45]); ax.axis('off')
        tb = ax.table(cellText=rows, colLabels=cols, loc='upper left', cellLoc='left', colLoc='left')
        tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.45)
        for (r, c), cell in tb.get_celld().items():
            cell.set_edgecolor(MUTED)
            if r == 0:
                cell.set_text_props(weight='bold')
        fig.text(0.07, 0.40, extra_text, fontsize=8.5, color=INK2, va='top', wrap=True, family='monospace')
        pdf.savefig(fig); plt.close(fig)

        # page 2: map, prediction coloured by position error
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        _track(ax)
        ax.plot(per['gx'], per['gy'], color=INK2, lw=1.0, label='GT')
        vmax = max(np.percentile(per['pos'], 95), 0.5)
        lc = _colored(ax, per['x'], per['y'], per['pos'], vmax)
        fig.colorbar(lc, ax=ax, label='position error [m]', shrink=0.6)
        ax.plot(per['x'][0], per['y'][0], 'o', color=INK, ms=8)
        ax.set_aspect('equal'); ax.legend(loc='upper left', frameon=False)
        ax.set_title('trajectory (prediction coloured by position error), dot = start')
        pdf.savefig(fig); plt.close(fig)

        # page 3: errors over time (one measure per axis, no dual axes)
        fig, axs = plt.subplots(4, 1, figsize=(11.69, 8.27), sharex=True)
        axs[0].plot(t, per['pos'], color=SERIES[0]); axs[0].set_ylabel('position err [m]')
        axs[1].plot(t, per['lat'], color=SERIES[0]); axs[1].set_ylabel('lateral err [m]')
        axs[2].plot(t, per['lon'], color=SERIES[0]); axs[2].set_ylabel('longit. err [m]')
        axs[3].plot(t, per['dyaw'], color=SERIES[0]); axs[3].set_ylabel('yaw err [deg]')
        axs[3].set_xlabel('time since run start [s]')
        for a in axs[1:]:
            a.axhline(0, color=MUTED, lw=0.8)
        fig.suptitle('error vs time (GT car frame)', color=INK)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # page 4: CDF of position error
        fig, ax = plt.subplots(figsize=(8.27, 5))
        e = np.sort(per['pos'])
        ax.plot(e, np.linspace(0, 100, len(e)), color=SERIES[0])
        for th in (0.25, 0.5, 1.0):
            ax.axvline(th, color=MUTED, lw=0.8, ls='--')
        ax.set_xscale('log'); ax.set_xlabel('position error [m]'); ax.set_ylabel('% of poses ≤ x')
        ax.set_title('cumulative position error')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def html_report(out_html, name, per, others=None):
    """Interactive map + linked error plot. others: {label: per-dict} for overlays."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    t = (per['ts'] - data.t0_ns()) / 1e9
    fig = make_subplots(rows=2, cols=1, row_heights=[0.72, 0.28], vertical_spacing=0.06,
                        subplot_titles=('map (hover for time / errors)', 'error vs time'))
    for k, v in data.track_bounds().items():
        fig.add_trace(go.Scattergl(x=v[:, 0], y=v[:, 1], mode='lines', line=dict(color=MUTED, width=1),
                                   name=k, hoverinfo='skip', showlegend=False), 1, 1)
    fig.add_trace(go.Scattergl(x=per['gx'], y=per['gy'], mode='lines', line=dict(color=INK2, width=2), name='GT',
                               text=[f't={a:.1f}s' for a in t], hoverinfo='text+name'), 1, 1)
    hov = [f't={a:.1f}s<br>pos err {p:.2f} m<br>lat {la:+.2f} lon {lo:+.2f}<br>yaw err {dy:+.2f}°'
           for a, p, la, lo, dy in zip(t, per['pos'], per['lat'], per['lon'], per['dyaw'])]
    fig.add_trace(go.Scattergl(x=per['x'], y=per['y'], mode='markers+lines', name=name,
                               line=dict(color=SERIES[0], width=1),
                               marker=dict(size=5, color=per['pos'], colorscale='Blues', cmin=0,
                                           cmax=float(max(np.percentile(per['pos'], 95), .5)),
                                           colorbar=dict(title='err m', len=0.6, y=0.7)),
                               text=hov, hoverinfo='text'), 1, 1)
    for j, (lab, o) in enumerate((others or {}).items()):
        to = (o['ts'] - data.t0_ns()) / 1e9
        fig.add_trace(go.Scattergl(x=o['x'], y=o['y'], mode='lines', name=lab,
                                   line=dict(color=SERIES[1 + j % 2], width=1.5)), 1, 1)
        fig.add_trace(go.Scattergl(x=to, y=o['pos'], mode='lines', name=f'{lab} err',
                                   line=dict(color=SERIES[1 + j % 2], width=1.5)), 2, 1)
    fig.add_trace(go.Scattergl(x=t, y=per['pos'], mode='lines', name='pos err', line=dict(color=SERIES[0])), 2, 1)
    fig.add_trace(go.Scattergl(x=t, y=np.abs(per['lat']), mode='lines', name='|lat err|',
                               line=dict(color=SERIES[0], dash='dot')), 2, 1)
    fig.update_yaxes(scaleanchor='x', scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text='t [s]', row=2, col=1)
    fig.update_yaxes(title_text='m', row=2, col=1)
    fig.update_layout(height=1100, title=name, template='plotly_white', hovermode='closest')
    fig.write_html(out_html, include_plotlyjs='cdn')


def debug_video(out_mp4, per, cam='fl', t_from=None, dur=30.0, win=60.0):
    """Side-by-side: camera frame | local top-down map (GT vs pred, ±win m), for visual debugging."""
    fr = data.frames(cam)
    t = (per['ts'] - data.t0_ns()) / 1e9
    t_from = t[0] if t_from is None else t_from
    idx = np.where((t >= t_from) & (t < t_from + dur))[0]
    W, H = 1506 // 2, 728 // 2
    S = H  # map panel size
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*'mp4v'), 10, (W + S, H))
    bounds = data.track_bounds()
    for i in idx:
        j = np.searchsorted(fr.stamp_ns.values, per['ts'][i]); j = min(j, len(fr) - 1)
        im = cv2.resize(cv2.imread(fr.path[j]), (W, H))
        cx, cy, th = per['gx'][i], per['gy'][i], per['gyaw'][i]
        sc = S / (2 * win)
        def to_px(X, Y):  # heading-up local map
            dx, dy = X - cx, Y - cy
            fx = np.cos(th) * dx + np.sin(th) * dy; ly = -np.sin(th) * dx + np.cos(th) * dy
            return np.stack([S / 2 - ly * sc, S * 0.75 - fx * sc], -1).astype(np.int32)
        mp = np.full((S, S, 3), 252, np.uint8)
        for v in bounds.values():
            near = np.hypot(v[:, 0] - cx, v[:, 1] - cy) < 2 * win
            if near.sum() > 1:
                cv2.polylines(mp, [to_px(v[near, 0], v[near, 1])], False, (178, 184, 185), 1)
        k0 = max(0, i - 100)
        cv2.polylines(mp, [to_px(per['gx'][k0:i + 1], per['gy'][k0:i + 1])], False, (78, 81, 82), 2)
        cv2.polylines(mp, [to_px(per['x'][k0:i + 1], per['y'][k0:i + 1])], False, (214, 120, 42), 2)
        for (X, Y, col) in [(per['gx'][i], per['gy'][i], (78, 81, 82)), (per['x'][i], per['y'][i], (214, 120, 42))]:
            cv2.circle(mp, tuple(int(a) for a in to_px(np.array([X]), np.array([Y]))[0]), 5, col, -1)
        txt = [f't {t[i]:.1f}s', f'err {per["pos"][i]:.2f} m', f'lat {per["lat"][i]:+.2f} lon {per["lon"][i]:+.2f}',
               f'yaw {per["dyaw"][i]:+.2f} deg', 'grey=GT blue=pred']
        for k, s in enumerate(txt):
            cv2.putText(mp, s, (8, 20 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (11, 11, 11), 1, cv2.LINE_AA)
        vw.write(np.hstack([im, mp]))
    vw.release()


def make_report(pred_path, name, split='val', video=False, extra=None, baselines=None, others=None):
    out = os.path.dirname(os.path.abspath(pred_path))
    pred = load_pred(pred_path)
    m, per = compute_all(pred, split)
    m['split'] = split
    if extra:
        m.update({k: v for k, v in extra.items() if isinstance(v, (int, float))})
    json.dump(m, open(f'{out}/metrics_{split}.json', 'w'), indent=1)
    bl = {}
    for lab, path in (baselines or {}).items():
        if os.path.exists(path):
            bl[lab] = json.load(open(path))
    txt = '\n'.join(f'{k}: {v}' for k, v in (extra or {}).items())
    pdf_report(f'{out}/report_{split}.pdf', name, m, per, txt, bl)
    oth = {}
    for lab, path in (others or {}).items():
        if os.path.exists(path):
            oth[lab] = compute_all(load_pred(path), split)[1]
    html_report(f'{out}/traj_{split}.html', name, per, oth)
    if video:
        debug_video(f'{out}/debug_{split}.mp4', per)
    return m


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pred'); ap.add_argument('--name', default='experiment')
    ap.add_argument('--split', default='val'); ap.add_argument('--video', action='store_true')
    a = ap.parse_args()
    m = make_report(a.pred, a.name, a.split, a.video)
    print(json.dumps(m, indent=1))
