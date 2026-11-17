"""Own task-1/2 episodes (b60f1e2e, local copy 0d3bf9f6 = eps 0-301) that are in training but have no box version, and why."""
import json, sys, collections
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, '.')
import yolocue_bridge as YC, v7_boxes as VB
V1 = Path.home() / 'Downloads/b60f1e2e_0d3bf9f6/v1'
meta = {m['episode_index']: m for m in pq.read_table(sorted((V1 / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()}
instr = {}
for f in ('instr_all.json', 'instr_0200.json', 'instr_0308.json'):
    instr.update({int(k): v for k, v in json.load(open(f)).items()})
EXCL = {193: 'demo moves another object', 257: 'papaya named mango', 282: 'wrong level', 294: 'wrong level'}
have = set()
for f in ('out/own_v7a_t1.json', 'out/own_v7a_t2.json', 'out/own0200_final_t1.json', 'out/own0308_v7a_t1.json', 'out/own0308_v7a_t2.json'):
    have |= {int(k) for k in json.load(open(f))['episodes']}
t = pq.read_table(sorted((V1 / 'data').glob('*/*.parquet'))[0], columns=['episode_index', 'frame_index', 'action.effector.position', 'observation.state.effector.position', 'observation.state.ee.position'])
ep, fi = t.column('episode_index').to_numpy(), t.column('frame_index').to_numpy()
C = np.array(t.column('action.effector.position').to_pylist(), float); G = np.array(t.column('observation.state.effector.position').to_pylist(), float); E = np.array(t.column('observation.state.ee.position').to_pylist(), float)
det = YC.detector(YC.load_config()); out = {}; cnt = collections.Counter()
for e in sorted(instr):
    if e in have or e in EXCL: continue
    m = np.where(ep == e)[0]; m = m[np.argsort(fi[m])]; c, g, ee = C[m], G[m], E[m]; gf = VB.grasp_frame(c)
    arms = [x for x in (0, 1) if (c[:, x] > YC.CMD_OPEN).any()]; why = []; hold = lift = None
    if gf is None: why.append('no command close')
    if len(arms) != 1: why.append(f'{len(arms)} arms')
    if gf is not None and len(arms) == 1:
        x = arms[0]; hold = round(float(np.median(g[gf + 15:gf + 30, x]) - np.median(g[:15, x])), 3); lift = round(float(ee[gf:gf + 60, 7 * x + 2].max() - ee[gf, 7 * x + 2]), 3)
        if not hold > 0.02: why.append('hold')
        if not lift > 0.03: why.append('lift')
    pr = YC.parse(instr[e]); rgb = VB.frame0(V1, meta[e]); yr = YC.select(det(rgb), *pr, rgb=rgb)[2]
    out[e] = {'instruction': instr[e], 'parsed': pr, 'grasp_frame': gf, 'arms': arms, 'hold': hold, 'lift': lift, 'rule': why, 'yolo': yr, 'T': int(len(m))}
    cnt[('rule ok' if not why else '+'.join(why), yr)] += 1
json.dump({str(k): v for k, v in out.items()}, open('out/extra_cands.json', 'w'), indent=1, ensure_ascii=False)
print('candidates', len(out)); [print(' ', k, v) for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])]
