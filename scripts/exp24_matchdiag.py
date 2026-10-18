"""Exp24 diagnosis: where does matching to the night map fail on day proxies? Stage-0 descriptor matches (n_m1)
vs PnP inliers, per run (online_info.csv)."""
import sys, numpy as np, pandas as pd
runs = sys.argv[1:]
for r in runs:
    d = pd.read_csv(f'{r}/online_info.csv')
    m1 = d.n_m1.fillna(0).values; inl = d.n_inl.fillna(0).values
    fix = d.status.isin(['fix', 'reloc']).values
    q = lambda a: '/'.join(f'{x:.0f}' for x in np.percentile(a, [10, 50, 90]))
    print(f'{r}\n  n={len(d)} fix {fix.mean()*100:.1f}%  n_m1 p10/50/90 {q(m1)}  n_m1<30 {np.mean(m1<30)*100:.1f}%'
          f'  m1>=30&nofix {np.mean((m1>=30)&~fix)*100:.1f}%  inl/m1 (fixes) {np.median(inl[fix]/np.maximum(m1[fix],1)):.2f}'
          f'  rms_px(fixes) {d.rms_px[fix].median():.2f}  status {d.status.value_counts().to_dict()}')
