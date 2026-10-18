"""Write outputs/20_map/train_ref.npz: the training-run columns that test mode reads, so inference needs only the
checkpoints and the test bag (no training bag / data/extracted). Shipped with the checkpoints (contains training GT:
share privately, dataset licence).

  gt:    timestamp_ns, x, y, yaw                 (relocalisation database poses, stand-in first pose)
  state: timestamp_ns, pos_accuracy.0, pos_accuracy.1   (map filter: frames with badly geo-referenced training GT)

    python scripts/build_train_ref.py          # needs data/extracted (scripts/extract_mcap.py on the training bag)
"""
import os
import sys

import numpy as np

sys.path.insert(0, '.')
from odo import data

COLS = dict(gt=['timestamp_ns', 'x', 'y', 'yaw'], state=['timestamp_ns', 'pos_accuracy.0', 'pos_accuracy.1'])

if __name__ == '__main__':
    src = dict(gt=data.gt(), state=data.state())
    z = {}
    for name, cols in COLS.items():
        z[f'{name}_cols'] = np.array(cols)
        for i, c in enumerate(cols):
            z[f'{name}_{i}'] = src[name][c].values
    os.makedirs(os.path.dirname(data.TRAIN_REF), exist_ok=True)
    np.savez_compressed(data.TRAIN_REF, **z)
    for name, cols in COLS.items():                 # the fallback must return exactly the CSV values
        ref = data._train_ref(name)
        for c in cols:
            assert np.array_equal(ref[c].values, src[name][c].values), (name, c)
    print(f'{data.TRAIN_REF}: gt {len(src["gt"])} rows, state {len(src["state"])} rows, '
          f'{os.path.getsize(data.TRAIN_REF) / 1e6:.1f} MB (identical to data/extracted)')
