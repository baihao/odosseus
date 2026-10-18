#!/usr/bin/env python3
"""Extract everything we need from the camloc mcap into plain files (one pass).

    python scripts/extract_mcap.py data/c0_.../c0_..._0.mcap data/extracted

Writes
  gt.csv                 timestamp_ns,x,y,z,yaw,qx,qy,qz,qw   (/tf map->base_link, GT, never an input)
  state.csv              flattened /constructor0/state_estimation (GT pose fields + allowed speed/yaw-rate)
  calib.json             per camera: K, D, width, height, T_base_cam (4x4, base_link <- camera optical)
  tf_static.json         all static transforms
  track_bounds.json      {ns: [[x,y,z],...]}
  images/<cam>/<stamp_ns>.jpg   raw jpeg bytes (no re-encoding)
  images/<cam>.csv       stamp_ns,log_time_ns,file
"""
import csv, json, math, os, sys
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

CAMS = ['fl', 'fr', 'cl', 'cr', 'r', 'rl', 'rr']
NS = 'constructor0'


def stamp(h):
    return h.stamp.sec * 10**9 + h.stamp.nanosec


def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def flatten(msg, prefix=''):
    """ROS message -> flat dict of scalars."""
    out = {}
    if hasattr(msg, '__slots__') or hasattr(msg, '__dict__'):
        names = getattr(msg, '__slots__', None) or list(vars(msg).keys())
        for n in names:
            v = getattr(msg, n)
            out.update(flatten(v, f'{prefix}{n}.' if not isinstance(v, (int, float, bool, str)) else f'{prefix}{n}'))
        return out
    if isinstance(msg, (list, tuple)) or (hasattr(msg, '__len__') and not isinstance(msg, (str, bytes))):
        if len(msg) <= 16:
            for i, v in enumerate(msg):
                out.update(flatten(v, f'{prefix}{i}' if isinstance(v, (int, float, bool)) else f'{prefix}{i}.'))
        return out
    key = prefix.rstrip('.')
    if isinstance(msg, (int, float, bool)):
        out[key] = msg
    elif isinstance(msg, str):
        out[key] = msg
    return out


def main(bag, outdir):
    os.makedirs(outdir, exist_ok=True)
    for c in CAMS:
        os.makedirs(f'{outdir}/images/{c}', exist_ok=True)
    img_topics = {f'/{NS}/sensor/camera_{c}/compressed_image': c for c in CAMS}
    info_topics = {f'/{NS}/sensor/camera_{c}/camera_info': c for c in CAMS}
    # the daytime bag (L9) has no state_estimation: its car state is /validation/twist + /accel (base_link)
    TWIST, ACCEL = f'/{NS}/validation/twist', f'/{NS}/validation/accel'
    topics = list(img_topics) + list(info_topics) + ['/tf', '/tf_static', f'/{NS}/state_estimation',
                                                     f'/{NS}/track_bounds', TWIST, ACCEL]
    tw_rows, ac_rows = [], []
    gt_f = open(f'{outdir}/gt.csv', 'w', newline=''); gt_w = csv.writer(gt_f)
    gt_w.writerow(['timestamp_ns', 'x', 'y', 'z', 'yaw', 'qx', 'qy', 'qz', 'qw'])
    idx_f = {c: open(f'{outdir}/images/{c}.csv', 'w', newline='') for c in CAMS}
    idx_w = {c: csv.writer(f) for c, f in idx_f.items()}
    for w in idx_w.values():
        w.writerow(['stamp_ns', 'log_time_ns', 'file'])
    state_rows, state_keys = [], None
    infos, tf_static, bounds = {}, {}, None
    n = 0
    with open(bag, 'rb') as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, ch, m, msg in r.iter_decoded_messages(topics=topics):
            t = ch.topic
            n += 1
            if n % 20000 == 0:
                print(n, t, flush=True)
            if t in img_topics:
                c = img_topics[t]; s = stamp(msg.header)
                fn = f'images/{c}/{s}.jpg'
                with open(f'{outdir}/{fn}', 'wb') as o:
                    o.write(bytes(msg.data))
                idx_w[c].writerow([s, m.log_time, fn])
            elif t in info_topics:
                c = info_topics[t]
                if c not in infos:
                    infos[c] = dict(K=list(msg.k), D=list(msg.d), P=list(msg.p), R=list(msg.r),
                                    width=msg.width, height=msg.height, model=msg.distortion_model,
                                    frame_id=msg.header.frame_id)
            elif t == '/tf':
                for tr in msg.transforms:
                    if tr.header.frame_id == 'map' and tr.child_frame_id == f'{NS}/base_link':
                        q, p = tr.transform.rotation, tr.transform.translation
                        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                        gt_w.writerow([stamp(tr.header), f'{p.x:.4f}', f'{p.y:.4f}', f'{p.z:.4f}', f'{yaw:.6f}',
                                       f'{q.x:.7f}', f'{q.y:.7f}', f'{q.z:.7f}', f'{q.w:.7f}'])
            elif t == '/tf_static':
                for tr in msg.transforms:
                    q, p = tr.transform.rotation, tr.transform.translation
                    tf_static[f'{tr.header.frame_id}->{tr.child_frame_id}'] = dict(
                        parent=tr.header.frame_id, child=tr.child_frame_id,
                        t=[p.x, p.y, p.z], q=[q.x, q.y, q.z, q.w])
            elif t == f'/{NS}/state_estimation':
                d = flatten(msg)
                d = {k: v for k, v in d.items() if not isinstance(v, str)}
                if state_keys is None:
                    state_keys = list(d.keys())
                state_rows.append([d.get(k, '') for k in state_keys])
            elif t == TWIST:
                tw_rows.append((stamp(msg.header), msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z))
            elif t == ACCEL:
                ac_rows.append((stamp(msg.header), msg.accel.linear.x, msg.accel.linear.y))
            elif t == f'/{NS}/track_bounds' and bounds is None:
                bounds = {}
                for mk in msg.markers:
                    bounds[mk.ns or str(mk.id)] = [[p.x, p.y, p.z] for p in mk.points]
    gt_f.close()
    for f in idx_f.values():
        f.close()
    if state_keys is None and tw_rows:
        # allowed motion fields only, in the state.csv layout (v_mps = |(vx, vy)|, accel interpolated to the twist stamps)
        tw = sorted(tw_rows); ac = np.array(sorted(ac_rows), float) if ac_rows else np.zeros((1, 3))
        state_keys = ['header.stamp.sec', 'header.stamp.nanosec', 'vx_mps', 'vy_mps', 'v_mps', 'wz_radps', 'ax_mps2', 'ay_mps2']
        for s, vx, vy, wz in tw:
            s = int(s)
            state_rows.append([s // 10**9, s % 10**9, vx, vy, math.hypot(vx, vy), wz,
                               float(np.interp(s, ac[:, 0], ac[:, 1])), float(np.interp(s, ac[:, 0], ac[:, 2]))])
    with open(f'{outdir}/state.csv', 'w', newline='') as o:
        w = csv.writer(o); w.writerow(state_keys); w.writerows(state_rows)
    json.dump(tf_static, open(f'{outdir}/tf_static.json', 'w'), indent=1)
    json.dump(bounds, open(f'{outdir}/track_bounds.json', 'w'))
    # calibration: chain static tfs base_link -> camera optical frame
    calib = {}
    for c, inf in infos.items():
        frame = inf['frame_id'] or f'{NS}/camera_{c}'
        T = chain(tf_static, f'{NS}/base_link', frame)
        calib[c] = dict(inf, T_base_cam=T.tolist() if T is not None else None)
    json.dump(calib, open(f'{outdir}/calib.json', 'w'), indent=1)
    print('done', n, 'msgs; state keys:', state_keys)


def chain(tfs, parent, child):
    """4x4 transform parent <- child by walking static tf edges (BFS, both directions)."""
    edges = {}
    for v in tfs.values():
        T = np.eye(4); T[:3, :3] = quat_to_R(*v['q']); T[:3, 3] = v['t']
        edges.setdefault(v['parent'], []).append((v['child'], T))
        edges.setdefault(v['child'], []).append((v['parent'], np.linalg.inv(T)))
    from collections import deque
    q = deque([(parent, np.eye(4))]); seen = {parent}
    while q:
        node, T = q.popleft()
        if node == child:
            return T
        for nb, Te in edges.get(node, []):
            if nb not in seen:
                seen.add(nb); q.append((nb, T @ Te))
    return None


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
