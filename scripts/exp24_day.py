"""Exp24: night -> day transfer. Runs the online localiser on one lap of one fold with the query images replaced by a
day proxy (scripts/day_translate.py) and a config override, then scores it.

    python scripts/exp24_day.py <name> <images: night|day|day_nolights> '<cfg json>' [--fold F5] [--lap L2] [--video]
Output: outputs/24_day/<images>/<name>/<fold>_<lap>/ (pred.csv, online_info.csv, metrics, report_<lap>.pdf,
run_report.pdf = error / innovation / fix-rate report, with the night run of the same config as grey reference)."""
import argparse
import json
import os
import sys

sys.path.insert(0, '.')
from odo import data, online

ROOTS = {'night': None, 'day': 'data/day_proxy/{lap}', 'day_nolights': 'data/day_proxy/{lap}_nolights',
         'day_qwen': 'data/day_proxy/{lap}_qwen', 'day_style': 'data/day_proxy/{lap}_style'}

ap = argparse.ArgumentParser()
ap.add_argument('name'); ap.add_argument('images', choices=list(ROOTS)); ap.add_argument('cfg', nargs='?', default='{}')
ap.add_argument('--fold', default='F5'); ap.add_argument('--lap', default='L2'); ap.add_argument('--video', action='store_true')
ap.add_argument('--limit', type=int)
ap.add_argument('--root', default='outputs/24_day', help='output root')
ap.add_argument('--cams', nargs='+', default=data.CAMS, help='cameras used for localisation (default all 7)')
a = ap.parse_args()
cfg = json.loads(a.cfg)
if ROOTS[a.images]:
    cfg['img_root'] = os.path.abspath(ROOTS[a.images].format(lap=a.lap))
if list(a.cams) != list(data.CAMS):         # camera subset: part of the run name (night reference must match)
    a.name = f'{a.name}_{"".join(a.cams)}'
out = f'{a.root}/{a.images}/{a.name}/{a.fold}_{a.lap}'
pred, inf, ms, stat = online.run(a.lap, out, cams=a.cams, fold=a.fold, cfg=cfg, verbose=True, limit=a.limit)
m = online.evaluate(out, a.lap, f'Exp24 {a.name}, {a.images} images, {a.fold}/{a.lap}', ms, stat, video=a.video,
                    extra=dict(fold=a.fold, images=a.images, cfg=json.dumps(cfg)))
from odo import run_report
ref = f'{a.root}/night/{a.name}/{a.fold}_{a.lap}'
run_report.make(out, f'Exp24 {a.name}, {a.images} images, {a.fold}/{a.lap}',
                ref=ref if a.images != 'night' and os.path.exists(f'{ref}/pred.csv') else None, ref_name='night images',
                meta={'fold / map laps': f'{a.fold} / {" ".join(data.FOLDS[a.fold]["map_laps"])} (night map)',
                      'query images': {'night': 'original night images', 'day': 'day proxy (CycleGAN-Turbo night_to_day)',
                                       'day_nolights': 'day proxy with light sources removed before translation',
                                       'day_qwen': 'day proxy by Qwen-Image-Edit-2509 + Lightning (midday sun, all lights off)',
                                       'day_style': 'day proxy by CAP-VSTNet photorealistic style transfer (lights removed)'}[a.images]})
print('RESULT', a.images, a.name, a.fold, a.lap, json.dumps({k: round(v, 4) for k, v in m.items() if isinstance(v, float)}),
      stat, 'n_inl_median', float(inf.n_inl.median()), flush=True)
