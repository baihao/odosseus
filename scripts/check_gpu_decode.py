import sys; sys.path.insert(0, '.')
import numpy as np, cv2, torch, time
from odo import data, online
fe = online.Frontend(data.CAMS)
sets = online.image_sets('tune', data.CAMS)
paths = list(sets.iloc[100][data.CAMS])
for g in [True, False, True, False]:
    fe.gpu_decode = g; torch.cuda.synchronize(); t = time.perf_counter()
    for k in range(20):
        ims = fe.load(list(sets.iloc[100 + k][data.CAMS])); f = fe.features(ims)
    torch.cuda.synchronize(); print('gpu_decode', g, '%.1f ms per set (load+SP)' % ((time.perf_counter() - t) / 20 * 1000))
fe.gpu_decode = True; a = fe.load(paths)[0].cpu().numpy(); b = cv2.imread(paths[0], 0)
print('max abs diff GPU vs CPU decode', np.abs(a.astype(int) - b).max(), 'mean', np.abs(a.astype(int) - b).mean())
