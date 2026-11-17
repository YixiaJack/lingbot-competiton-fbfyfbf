"""Batches 27a/27b = b60f1e2e 42da9074 (01:35) eps 613-616 (task 1) + cfa641b0 2ffa8d81 (01:53) eps 20-23 (task 3):
normal teleop demo or DAgger recording? (dagger_seg.segments: None = pure human demo)"""
import json, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, '.'); import dagger_seg as DS
out = {}
for tag, V1, eps in (('42da', Path.home() / 'Downloads/b60f1e2e_42da9074/v1', range(613, 617)), ('2ffa', Path.home() / 'Downloads/cfa641b0_2ffa8d81/v1', range(20, 24))):
    lab = {}
    for l in open(V1 / 'meta/labels.jsonl'):
        r = json.loads(l); L = (r.get('labels') or [{}])[0]; lab[r['episode_index']] = (L.get('task_instruction') or '', L.get('result'))
    cols = ['episode_index', 'frame_index', 'action.effector.position', 'action.arm.position', 'observation.state.arm.position']
    ts = [pq.read_table(f, columns=cols) for f in sorted((V1 / 'data').glob('*/*.parquet'))]
    import pyarrow as pa; t = pa.concat_tables(ts)
    ep, fi = t.column('episode_index').to_numpy(), t.column('frame_index').to_numpy()
    C, A, S = (np.array(t.column(c).to_pylist(), float) for c in cols[2:])
    out[tag] = {}
    for e in eps:
        m = np.where(ep == e)[0]; m = m[np.argsort(fi[m])]; segs, rm, gp = DS.segments(C[m], A[m], S[m])
        out[tag][str(e)] = None if segs is None else [list(map(int, s)) for s in segs]
        print(f"{tag} ep{e} T{len(m)} {lab[e][1]} {'DEMO' if segs is None else 'DAgger ' + str(out[tag][str(e)])} | grip cmd [{C[m].min():.2f}, {C[m].max():.2f}] | {lab[e][0][:60]}")
json.dump(out, open('out/classify_two.json', 'w'))
json.dump({k: v for k, v in out['42da'].items()}, open('out/classify_42da.json', 'w'))
