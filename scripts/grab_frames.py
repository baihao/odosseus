"""Save frames of an mp4 as PNGs: python scripts/grab_frames.py <mp4> <out_prefix> <frame> [<frame> ...]"""
import sys, cv2
cap = cv2.VideoCapture(sys.argv[1])
for f in map(int, sys.argv[3:]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
    if ok:
        cv2.imwrite(f'{sys.argv[2]}_{f}.png', im)
print('GRAB_DONE')
