"""Recompute the task-3 split of a private_prep JSON with the corrected real_split.segment (same code as
private_prep.py's task-3 branch); every other field is kept as is. usage: fix_prep.py V1_DIR IN_JSON OUT_JSON"""
import json, sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import real_split as R

d, src, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
J = json.loads(src.read_text()); P = J['episodes']
t = pa.concat_tables([pq.read_table(f, columns=['episode_index', 'frame_index', 'observation.state.effector.position', 'observation.state.ee.position'])
                      for f in sorted((d / 'data').glob('*/*.parquet'))])
ep = t.column('episode_index').to_numpy(); fi = t.column('frame_index').to_numpy()
G = np.array(t.column('observation.state.effector.position').to_pylist()); E = np.array(t.column('observation.state.ee.position').to_pylist())
changed = []
for k, rec in P.items():
    if rec['task'] != 3: continue
    e = int(k); m = ep == e; o = np.argsort(fi[m]); g, ee = G[m][o], E[m][o]
    before = json.dumps({x: rec.get(x) for x in ('split', 'segments', 'use_t3_split', 'split_error')}, sort_keys=True)
    for x in ('split', 'pours', 'segments', 'split_error'): rec.pop(x, None)
    try:
        s = R.segment(g, ee)
        pan = 0 if s['pan_arm'] == 'left' else 1
        x = ee[:, 7 * pan]; x0 = np.median(x[s['b1']:s['b2']])
        runs = [(u, v) for u, v in R.runs(x >= x0 + 0.5 * (s['x_pour'] - x0)) if v - u >= 15]
        merged = []
        for u, v in runs:
            if merged and u - merged[-1][1] < 30: merged[-1] = (merged[-1][0], v)
            else: merged.append((u, v))
        pours = sum(1 for u, v in merged if u < s['release'])
        rec.update(split=s, pours=pours, use_t3_split=(pours == 1))
        if pours == 1:
            b = [0, s['b1'], s['b2'], s['b3'], s['T']]
            rec['segments'] = [{'subtask': st, 'start': b[i], 'end': b[i + 1], 'instruction': R.instruction(st, s['pan_arm'])}
                               for i, st in enumerate(R.SUBTASK_IDS)]
    except ValueError as ex:
        rec.update(split_error=str(ex), use_t3_split=False)
    after = json.dumps({x: rec.get(x) for x in ('split', 'segments', 'use_t3_split', 'split_error')}, sort_keys=True)
    if after != before: changed.append((k, rec.get('use_t3_split'), rec.get('split', {}).get('pan_arm'), [(s_['subtask'], s_['start'], s_['end']) for s_ in rec.get('segments', [])]))
J['summary']['t3_split_ok'] = sum(r.get('use_t3_split', False) for r in P.values() if r['task'] == 3)
dst.write_text(json.dumps(J, indent=1, ensure_ascii=False))
print(f'{src.name}: task-3 episodes changed {len(changed)}; split ok {J["summary"]["t3_split_ok"]}')
for c in changed: print('  ', c)
