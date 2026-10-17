"""Overlay the surveyed track borders (projected with the GT 6-DoF pose) on all 7 cameras: checks what the borders
correspond to in the image (painted line / kerb / wall) and how well calibration + GT agree."""
import sys; sys.path.insert(0, '.')
import cv2, numpy as np
from odo import data
from odo.reloc_pnp import GT6
from odo.map_build import DT_NS
gt6 = GT6(); tb = data.track_bounds(); cal = data.calib(); t0 = data.t0_ns()
cols = {'track_left': (0, 255, 0), 'track_right': (0, 0, 255), 'pitlane_left': (255, 255, 0), 'pitlane_right': (255, 0, 255)}
for tt in [float(a) for a in sys.argv[1:]] or [700.0]:
    ims = []
    for c in data.CAMS:
        f = data.frames(c); i = np.argmin(np.abs(f.stamp_ns.values - (t0 + tt * 1e9)))
        im = cv2.imread(f.path[i]); K, D, Tbc = cal[c]['K'], cal[c]['D'], cal[c]['T_base_cam']
        Tcm = np.linalg.inv(gt6.T(f.stamp_ns.values[i] + DT_NS)[0] @ Tbc)
        for k, P in tb.items():
            Xc = P @ Tcm[:3, :3].T + Tcm[:3, 3]
            ok = (Xc[:, 2] > 1.0) & (Xc[:, 2] < 80)
            n = Xc[:, :2] / Xc[:, 2:3]
            ok &= np.linalg.norm(n, axis=1) < 1.6
            if ok.sum() < 2: continue
            uv, _ = cv2.projectPoints(Xc[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
            uv = uv.reshape(-1, 2)
            for a, b in zip(uv[:-1], uv[1:]):
                if np.linalg.norm(a - b) < 200:
                    cv2.line(im, tuple(np.int32(a)), tuple(np.int32(b)), cols[k], 2)
        cv2.putText(im, c, (20, 60), 0, 2, (0, 255, 255), 3)
        ims.append(cv2.resize(im, (753, 364)))
    ims.append(np.zeros_like(ims[0]))
    cv2.imwrite(f'outputs/22_borders/overlay_{tt:.0f}.jpg', np.vstack([np.hstack(ims[i:i + 2]) for i in range(0, 8, 2)]))
