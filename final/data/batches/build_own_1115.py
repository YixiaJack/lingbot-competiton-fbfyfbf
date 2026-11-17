"""Own-data datasets from b60f1e2e cbd7865e (local copy: data, meta, camera_top only; on the dev machine the videos dir
is relinked to the full snapshot copy):
  own_t1_box_v7a / own_t2_box_v7a  box cues by YOLO v7a (all 131 checked by eye), instructions = training text (old) or
                                   official-distribution rewrite (new); built by derive_box_dataset.py
  own_t12_new1115                  plain copies of the 39 new task-1/2 episodes with the rewritten prompts
  own_t3_sub_new1115               the new task-3 episodes split into the 4 subtasks (real_split.segment); ep95 excluded
                                   (two pours, audit_1114)"""
import json, subprocess, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import derive_dataset as DD
sys.path.insert(0, '.')
import real_split as R
V1 = Path.home() / 'Downloads/b60f1e2e_cbd7865e/v1'; O = Path('out/ds_own'); O.mkdir(parents=True, exist_ok=True)
fps = json.loads((V1 / 'meta/info.json').read_text())['fps']
instr = json.load(open('instr_all.json'))
Path('instr_all_dbd.json').write_text(json.dumps({'episodes': {k: {'instruction': v} for k, v in instr.items()}}, ensure_ascii=False))
py = str(Path.home() / 'lingbot_v7env/bin/python')
for t in (1, 2):
    out = O / f'own_t{t}_box_v7a'
    subprocess.run(['rm', '-rf', str(out)], check=True)
    r = subprocess.run([py, '-I', str(Path.home() / 'lingbot-final-notes/experiments/autolabel/derive_box_dataset.py'), '--src', str(V1), '--videos', str(V1 / 'videos'),
                        '--boxes', f'out/own_v7a_t{t}.json', '--out', str(out), '--workers', '8', '--instructions', 'instr_all_dbd.json'], capture_output=True, text=True)
    print(f'own_t{t}_box_v7a', r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-800:])
new = json.load(open('new_t12.json'))
items = [{'src_ep': int(k), 'start': 0, 'end': None, 'task': v} for k, v in sorted(new.items(), key=lambda kv: int(kv[0]))]
_, eps, _ = DD.load_src(V1)
for it in items: it['end'] = eps[it['src_ep']]['length']
print('own_t12_new1115', DD.build(V1, O / 'own_t12_new1115', items, fps))
t3 = [e for e in json.load(open('new_t3.json')) if e != 95]
t = pq.read_table(sorted((V1 / 'data').glob('*/*.parquet'))[0], columns=['episode_index', 'frame_index', 'observation.state.effector.position', 'observation.state.ee.position'])
ep, fi = t.column('episode_index').to_numpy(), t.column('frame_index').to_numpy()
G = np.array(t.column('observation.state.effector.position').to_pylist(), float); E = np.array(t.column('observation.state.ee.position').to_pylist(), float)
items3, failed, recs = [], [], []
for e in t3:
    m = np.where(ep == e)[0]; m = m[np.argsort(fi[m])]
    try:
        s = R.segment(G[m], E[m])
    except ValueError as ex:
        failed.append((e, str(ex))); continue
    b = [0, s['b1'], s['b2'], s['b3'], s['T']]
    for i, k in enumerate(R.SUBTASK_IDS):
        items3.append({'src_ep': e, 'start': b[i], 'end': b[i + 1], 'task': R.instruction(k, s['pan_arm']), 'subtask': k, 'pan_arm': s['pan_arm']})
    recs.append((e, s['pan_arm'], [round((b[i + 1] - b[i]) / fps, 1) for i in range(4)]))
print('own_t3_sub_new1115', DD.build(V1, O / 'own_t3_sub_new1115', items3, fps), '| split', len(recs), 'failed', failed)
for r in recs: print('   ', r)
