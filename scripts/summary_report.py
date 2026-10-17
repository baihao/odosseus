"""Collect experiments into one comparison PDF + interactive HTML.

    python scripts/summary_report.py stage1   |   python scripts/summary_report.py stage2
"""
import json, os, sys
sys.path.insert(0, '.')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from odo import report, data

STAGES = {
    'stage1': dict(title='Stage 1 - visual odometry, val laps (t = 897..1120 s)', out='outputs/stage1_summary', exps=[
        ('dead reckoning (car state, ref.)', 'outputs/01_deadreckoning/pred_val.csv'),
        ('Exp03 Pi-Long fr (heading)', 'outputs/03_pi_long_fr/eval/heading/pred.csv'),
        ('Exp04 essential VO 7 cams', 'outputs/04_feature_vo_sp/fuse_val/pred.csv'),
        ('Exp05 planar+essential VO 7 cams', 'outputs/05_planar_vo/fuse_val/pred.csv'),
        ('Exp06 gyro + VO bias', 'outputs/06_vo_gyro/pred.csv'),
    ], notes=[
        'Exp02 MASt3R-SLAM: tracking lost after 1-3 frames on every camera -> no trajectory (see outputs/02_mast3r_slam).',
        'All VO runs: metric step length from car-state speed, anchored at the GT pose of the first val frame.',
        'VO methods use only cameras (+speed for scale); dead reckoning and Exp06 also use the car-state yaw rate.',
    ]),
    'stage2': dict(title='Stage 2 - image retrieval relocalisation, val laps (db = training run)', out='outputs/stage2_summary',
                   exps=[], notes=[]),
}
KEYS = [('pos_rmse', 'pos RMSE [m]'), ('lat_rmse', 'lat RMSE [m]'), ('lon_rmse', 'lon RMSE [m]'), ('yaw_rmse_deg', 'yaw RMSE [°]'),
        ('pos_median', 'median [m]'), ('within_100', '≤1 m [%]'), ('rpe_1s_yaw_rmse_deg', 'RPE1s yaw [°]'),
        ('rpe_10s_trans_pct', 'RPE10s drift [%]'), ('rpe_10s_yaw_rmse_deg', 'RPE10s yaw [°]')]


def main(stage):
    cfg = STAGES[stage]
    if stage == 'stage2':
        cfg['exps'] = json.load(open('outputs/stage2_summary/exps.json'))
        cfg['notes'] = json.load(open('outputs/stage2_summary/notes.json'))
    os.makedirs(cfg['out'], exist_ok=True)
    rows, pers = [], {}
    for name, path in cfg['exps']:
        if not os.path.exists(path):
            print('missing', path); continue
        m, per = report.compute_all(report.load_pred(path), 'val')
        rows.append([name] + [m.get(k, np.nan) for k, _ in KEYS]); pers[name] = per
    df = pd.DataFrame(rows, columns=['method'] + [l for _, l in KEYS])
    df.to_csv(f'{cfg["out"]}/metrics.csv', index=False)
    with PdfPages(f'{cfg["out"]}/summary.pdf') as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.04, 0.94, cfg['title'], fontsize=14, weight='bold')
        ax = fig.add_axes([0.03, 0.35, 0.94, 0.55]); ax.axis('off')
        cell = [[r[0]] + [f'{v:.3f}' if abs(v) < 10 else f'{v:.1f}' for v in r[1:]] for r in df.values.tolist()]
        tb = ax.table(cellText=cell, colLabels=df.columns, loc='upper left', cellLoc='left', colLoc='left',
                      colWidths=[0.28] + [0.08] * (len(df.columns) - 1))
        tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.6)
        for (r, c), ce in tb.get_celld().items():
            ce.set_edgecolor(report.MUTED)
            if r == 0: ce.set_text_props(weight='bold')
        fig.text(0.04, 0.3, '\n'.join(cfg['notes']), fontsize=9, va='top', color=report.INK2)
        pdf.savefig(fig); plt.close(fig)
        # map overlay, one panel per method (small multiples: no colour-only identity)
        n = len(pers)
        fig, axs = plt.subplots(1, n, figsize=(4 * n, 6), squeeze=False)
        for ax, (name, per) in zip(axs[0], pers.items()):
            report._track(ax)
            ax.plot(per['gx'], per['gy'], color=report.INK2, lw=1, label='GT')
            ax.plot(per['x'], per['y'], color=report.SERIES[0], lw=1, label='pred')
            ax.set_aspect('equal'); ax.set_title(name, fontsize=8); ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # position error vs time, small multiples on a shared log axis
        fig, axs = plt.subplots(n, 1, figsize=(11.69, 1.8 * n), sharex=True, sharey=True, squeeze=False)
        for ax, (name, per) in zip(axs[:, 0], pers.items()):
            t = (per['ts'] - data.t0_ns()) / 1e9
            ax.semilogy(t, np.maximum(per['pos'], 1e-2), color=report.SERIES[0], lw=1)
            ax.set_ylabel('pos err [m]', fontsize=7); ax.set_title(name, fontsize=8, loc='left')
        axs[-1, 0].set_xlabel('t [s]'); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    # interactive overlay
    first = list(pers)[0]
    others = {k: v for k, v in list(pers.items())[1:]}
    import plotly.graph_objects as go
    fig = go.Figure()
    for k, v in data.track_bounds().items():
        fig.add_trace(go.Scattergl(x=v[:, 0], y=v[:, 1], mode='lines', line=dict(color=report.MUTED, width=1), showlegend=False, hoverinfo='skip'))
    p0 = pers[first]
    fig.add_trace(go.Scattergl(x=p0['gx'], y=p0['gy'], mode='lines', name='GT', line=dict(color=report.INK2, width=2)))
    pal = report.SERIES + ['#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
    for j, (name, per) in enumerate(pers.items()):
        t = (per['ts'] - data.t0_ns()) / 1e9
        fig.add_trace(go.Scattergl(x=per['x'], y=per['y'], mode='lines', name=name, line=dict(color=pal[j % len(pal)], width=1.5),
                                   text=[f'{name}<br>t={a:.1f}s err {e:.2f} m' for a, e in zip(t, per['pos'])], hoverinfo='text'))
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    fig.update_layout(title=cfg['title'] + ' (click legend to toggle)', template='plotly_white', height=900)
    fig.write_html(f'{cfg["out"]}/overlay.html', include_plotlyjs='cdn')
    print(df.round(3).to_string())


if __name__ == '__main__':
    main(sys.argv[1])
