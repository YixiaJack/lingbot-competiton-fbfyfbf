"""Batch 24 = b60f1e2e c89c4a87 (22:22), eps 529-543 (task 1: 529-532, task 3: 533-543): normal teleop demo or DAgger recording?
dagger_seg.segments (jerk + gripper-command range, validated on fb04 333-339 and pure-policy 105a2d09 172-181) returns None
for a pure human demo, else the human segments. Prints one line per episode."""
import json, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, '.')
import dagger_seg as DS
V1 = Path.home() / 'Downloads/b60f1e2e_c89c4a87/v1'
lab = {}
for l in open(V1 / 'meta/labels.jsonl'):
    r = json.loads(l); L = (r.get('labels') or [{}])[0]; lab[r['episode_index']] = (L.get('task_instruction') or '', L.get('result'))
cols = ['episode_index', 'frame_index', 'action.effector.position', 'action.arm.position', 'observation.state.arm.position']
t = pq.read_table(sorted((V1 / 'data').glob('*/*.parquet'))[0], columns=cols)
ep, fi = t.column('episode_index').to_numpy(), t.column('frame_index').to_numpy()
C, A, S = (np.array(t.column(c).to_pylist(), float) for c in cols[2:])
out = {}
for e in range(529, 544):
    m = np.where(ep == e)[0]; m = m[np.argsort(fi[m])]
    segs, jerk, info = DS.segments(C[m], A[m], S[m])
    j = np.abs(np.diff(A[m], 2, axis=0)).sum(1)
    out[e] = None if segs is None else [list(map(int, s)) for s in segs]
    print(f"ep{e} T{len(m)} {lab[e][1]} {'DEMO' if segs is None else 'DAgger ' + str(out[e])} | jerk median {np.median(j):.4f} p90 {np.percentile(j, 90):.4f} "
          f"grip cmd [{C[m].min():.2f}, {C[m].max():.2f}] | {lab[e][0][:70]}")
json.dump({str(k): v for k, v in out.items()}, open('out/classify_c89c.json', 'w'))
