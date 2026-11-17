"""For every episode of the newest own snapshot (cbd7865e): which v4n training datasets already hold it (state
fingerprint) and the instruction those datasets train with. Read-only. usage: python3 own_map.py OUT.json"""
import glob, hashlib, json, sys, collections
import numpy as np, pyarrow.parquet as pq
def fps(d):
    t = pq.read_table(sorted(glob.glob(f'{d}/data/*/*.parquet'))[0], columns=['episode_index', 'frame_index', 'observation.state.arm.position', 'observation.state.effector.position'])
    ep, fi = t.column('episode_index').to_numpy(), t.column('frame_index').to_numpy()
    arm = np.array(t.column('observation.state.arm.position').to_pylist(), float); eff = np.array(t.column('observation.state.effector.position').to_pylist(), float)
    out = {}
    for e in np.unique(ep):
        m = np.where(ep == e)[0]; m = m[np.argsort(fi[m])][:30]
        out[int(e)] = hashlib.sha256(np.round(np.hstack([arm[m], eff[m]]), 4).tobytes()).hexdigest()[:16]
    return out
def tasks(d):
    return {int(m['episode_index']): (m.get('tasks') or [''])[0] for m in pq.read_table(sorted(glob.glob(f'{d}/meta/episodes/*/*.parquet'))[0]).to_pylist()}
held = collections.defaultdict(dict)
for l in open('/home/<dev-user>/lingbot_train/lists/v4n.txt'):
    p = l.split()
    if len(p) != 2 or 'public' in p[1]: continue
    name = p[1].rsplit('/', 1)[1]
    if name in {k for v in held.values() for k in v}: continue
    T = tasks(p[1])
    for e, h in fps(p[1]).items(): held[h][name] = T.get(e)
B = '/mnt/forge/datasets/private/b60f1e2e-b756-41e8-9fb4-e6b03679e371/snapshots/cbd7865e-e383-43d2-a5dd-ea85b5ec5b53/v1'
out = {e: held.get(h, {}) for e, h in fps(B).items()}
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print('episodes', len(out), 'already in training', sum(1 for v in out.values() if v), collections.Counter(tuple(sorted(v)) for v in out.values() if v).most_common(8))
