"""Side-by-side of the same frames: night original | day proxies (one column per proxy folder).
    python scripts/compare_proxies.py <out.jpg> <proxy_dir> [<proxy_dir> ...]
Frames: every file present in the first proxy folder (all cameras), up to 6 rows."""
import os
import sys

import cv2
import numpy as np

out, dirs = sys.argv[1], sys.argv[2:]
rows = []
for cam in ['fl', 'fr', 'cl', 'cr', 'r', 'rl', 'rr']:
    d = f'{dirs[0]}/images/{cam}'
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d))[:2]:
        ims = [cv2.imread(f'data/extracted/images/{cam}/{f}')] + [cv2.imread(f'{p}/images/{cam}/{f}') for p in dirs]
        rows.append(np.hstack([cv2.resize(i, (602, 291)) if i is not None else np.zeros((291, 602, 3), np.uint8) for i in ims]))
cv2.imwrite(out, np.vstack(rows[:6]), [cv2.IMWRITE_JPEG_QUALITY, 90])
print('saved', out, len(rows), 'rows')
