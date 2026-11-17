"""Extract per-arm wrist-camera crops of task-3 episodes with automatic grasp labels from the gripper trace.
usage: extract.py DATASET_V1 OUT_PREFIX [episodes: a-b,c,...|t3] [stride=15]
Label per arm and frame (gripper in 0-1 opening units, 30 fps):
  1 = holding: the gripper opened (>= 0.5) earlier and now sits steady (range <= 0.02 over 0.5 s) in 0.17-0.40
      (dustpan ~0.32, brush ~0.23).
  0 = not holding: reading >= 0.5 (open, or stuck halfway e.g. fingertips on the table); or <= 0.08 before the first
      opening (closed empty at the start) or after the last opening when that opening is a release (the gripper
      sat in the holding band before it; 11/16: ep173 right held the brush at 0.07 and never reopened).
 -1 = ambiguous, not used: near 0 between the first and the last opening (thin handle, rim grasp or empty),
      0.08-0.17, 0.40-0.5, or moving.
Writes OUT_PREFIX_X.npy (N, 96, 128, 3 uint8) and OUT_PREFIX_meta.npy (N, 6 float32: label, arm, episode, frame,
gripper, t)."""
import sys
from multiprocessing import Pool
from pathlib import Path
import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

D, OUT = Path(sys.argv[1]), sys.argv[2]
SEL = sys.argv[3] if len(sys.argv) > 3 else 't3'
STRIDE = int(sys.argv[4]) if len(sys.argv) > 4 else 15
ALL = len(sys.argv) > 5 and sys.argv[5] == 'all'   # keep ambiguous frames too (label -1), for timelines
W, H = 128, 96
CAMS = ('observation.images.camera_wrist_left', 'observation.images.camera_wrist_right')
META = {r['episode_index']: r for f in sorted((D / 'meta/episodes').glob('*/*.parquet')) for r in pq.read_table(f).to_pylist()}


def episodes():
    if SEL == 't3':
        return sorted(e for e, r in META.items() if any('dustpan' in str(x) for x in (r.get('tasks') or [])))
    out = []
    for part in SEL.split(','):
        if '-' in part:
            a, b = map(int, part.split('-')); out += list(range(a, b + 1))
        else:
            out.append(int(part))
    return [e for e in out if e in META]


def labels(g):
    T = len(g)
    lab = np.full(T, -1, np.int8)
    opened = np.where(g >= 0.5)[0]
    first_open = opened[0] if len(opened) else T
    last_open = opened[-1] if len(opened) else -1
    # the last opening is a release only if the gripper held something (sat in the holding band) before it
    held_before = last_open > 0 and bool(((g[:last_open] >= 0.17) & (g[:last_open] <= 0.40)).sum() >= 30)
    if not held_before:
        last_open = T
    for i in range(T):
        v = g[i]
        w = g[max(0, i - 15):i + 1]
        steady = len(w) >= 8 and w.max() - w.min() <= 0.02
        if v >= 0.5:
            lab[i] = 0
        elif v <= 0.08 and steady and (i < first_open or i > last_open):
            lab[i] = 0
        elif 0.17 <= v <= 0.40 and steady and i > first_open:
            lab[i] = 1
    return lab


def grippers(eps):
    out = {e: [] for e in eps}
    for f in sorted((D / 'data').glob('*/*.parquet')):
        t = pq.read_table(f, columns=['episode_index', 'frame_index', 'observation.state.effector.position'])
        ep = t.column('episode_index').to_numpy(); keep = np.isin(ep, eps)
        if not keep.any():
            continue
        fi = t.column('frame_index').to_numpy()[keep]
        g = np.array(t.column('observation.state.effector.position').to_pylist(), np.float32)[keep]
        for e in np.unique(ep[keep]):
            m = ep[keep] == e; out[int(e)].append((fi[m], g[m]))
    res = {}
    for e, parts in out.items():
        if not parts:
            continue
        fi = np.concatenate([p[0] for p in parts]); g = np.concatenate([p[1] for p in parts])
        res[e] = g[np.argsort(fi)]
    return res


def decode(e, cam, idx):
    m = META[e]
    p = D / 'videos' / cam / f"chunk-{m[f'videos/{cam}/chunk_index']:03d}" / f"file-{m[f'videos/{cam}/file_index']:03d}.mp4"
    t0 = m[f'videos/{cam}/from_timestamp']
    out, k = {}, 0
    with av.open(str(p)) as c:
        s = c.streams.video[0]
        c.seek(int(max(t0 + idx[0] / 30 - 1, 0) / s.time_base), stream=s, backward=True)
        for fr in c.decode(s):
            ts = float(fr.pts * s.time_base)
            while k < len(idx) and ts + 1e-3 >= t0 + idx[k] / 30:
                out[idx[k]] = np.asarray(fr.to_image().resize((W, H), Image.BILINEAR)); k += 1
            if k == len(idx):
                break
    return out


def job(args):
    e, g = args
    X, M = [], []
    for arm in (0, 1):
        lab = labels(g[:, arm])
        idx = [i for i in range(0, len(g), STRIDE) if lab[i] >= 0 or ALL]
        if not idx:
            continue
        try:
            fr = decode(e, CAMS[arm], idx)
        except Exception as ex:
            print('decode failed', e, arm, ex, flush=True); continue
        for i in idx:
            if i in fr:
                X.append(fr[i]); M.append((lab[i], arm, e, i, g[i, arm], i / 30))
    return (np.stack(X) if X else np.zeros((0, H, W, 3), np.uint8)), np.array(M, np.float32).reshape(-1, 6)


if __name__ == '__main__':
    eps = episodes()
    G = grippers(eps)
    print(f'{len(G)} episodes, stride {STRIDE}', flush=True)
    with Pool(8) as pool:
        res = pool.map(job, sorted(G.items()), chunksize=2)
    X = np.concatenate([r[0] for r in res]); M = np.concatenate([r[1] for r in res])
    np.save(OUT + '_X.npy', X); np.save(OUT + '_meta.npy', M)
    print('saved', X.shape, '| holding', int((M[:, 0] == 1).sum()), 'not holding', int((M[:, 0] == 0).sum()))
