"""Exp21 3-fold cross-validation (folds: odo.data.FOLDS): for a fold, run the online localiser on its two validation laps and on
the tuning lap L5, each anchored at its own first pose, with map = the fold's map laps. Parameters: odo.online.DEFAULT.
    python scripts/exp21_cv.py F1 [--video]"""
import sys, json; sys.path.insert(0, '.')
from odo import data, online
fold = sys.argv[1]; video = '--video' in sys.argv
laps = [data.tune_lap(fold)] if '--tune_only' in sys.argv else data.FOLDS[fold]['val'] + [data.tune_lap(fold)]
for lap in laps:
    out = f'outputs/21_online/cv/{fold}/{lap}'
    pred, inf, ms, stat = online.run(lap, out, fold=fold, verbose=False)
    role = 'tuning' if lap == data.tune_lap(fold) else 'validation'
    m = online.evaluate(out, lap, f'Exp21 {fold} ({data.FOLDS[fold]["map"]}), {role} lap {lap}', ms, stat, video=video and role == 'validation',
                        extra=dict(fold=fold, map_laps=' '.join(data.FOLDS[fold]['map_laps']), role=role))
    print('CV', fold, lap, role, json.dumps({k: round(v, 4) for k, v in m.items() if isinstance(v, float)}), stat, flush=True)
print('CV_DONE', fold)
