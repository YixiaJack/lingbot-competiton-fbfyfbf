"""Build own_t3_dagger_908a: task-3 DAgger recordings 466-474 of b60f1e2e 908aa6e5 (harness sessions with a human takeover),
HUMAN TAKEOVER SEGMENTS ONLY, cut at the subtask boundaries of the whole episode and labelled with that subtask's fixed
instruction (real_split.instruction; the strings the harness sends). One dataset; source_map.json carries 'subtask' per
episode so it can be split into own_t3_dagger_908a__{grasp,sweep,dump,home} like the t3_sub sets.
Pieces from dagger_t3_908a.py (out/dagger_t3_908a.json, checked by eye) plus ep469, whose split needed open_thr=0.34 (the
brush release only opened the right gripper to 0.36): b1 495, b2 996, b3 1309, pan left (checked against the frames:
lift starts ~f995, pan over the basket f1295, back down f1355)."""
import json, sys
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import derive_dataset as DD, real_split as R
V1 = Path.home() / 'Downloads/b60f1e2e_908aa6e5/v1'; O = Path('out/ds_own908a'); fps = 30
P = json.load(open('out/dagger_t3_908a.json'))
b469, human469 = [0, 495, 996, 1309, 1858], [(515, 1568), (1604, 1809)]
for st, en in human469:
    for k, sub in enumerate(R.SUBTASK_IDS):
        a, b = max(st, b469[k]), min(en, b469[k + 1])
        if b - a >= fps:
            P[f'469_{sub}_{a}'] = {'ep': 469, 'start': a, 'end': b, 'subtask': sub, 'pan_arm': 'left', 'secs': round((b - a) / fps, 1),
                                   'task': R.instruction(sub, 'left'), 'result': '成功', 'note': 'split with open_thr=0.34'}
P = dict(sorted(P.items(), key=lambda kv: (kv[1]['ep'], kv[1]['start'])))
final = {'note': 'task-3 DAgger 466-474, human takeover segments only, cut at subtask boundaries; all 9 end states checked by eye '
                 '(every paper ball in the basket, tools down); partial-success labels come from the policy part before the takeover',
         'pieces': P}
json.dump(final, open('out/dagger_t3_908a_final.json', 'w'), indent=1, ensure_ascii=False)
for k, v in P.items():
    print(f"  {k:18s} {v['subtask']:5s} {v['start']}-{v['end']} {v['secs']}s pan {v['pan_arm']} | {v['task'][:60]}")
items = [{'src_ep': v['ep'], 'start': v['start'], 'end': v['end'], 'task': v['task'], 'subtask': v['subtask'], 'pan_arm': v['pan_arm']} for v in P.values()]
print('own_t3_dagger_908a', DD.build(V1, O / 'own_t3_dagger_908a', items, fps),
      {s: sum(1 for v in P.values() if v['subtask'] == s) for s in R.SUBTASK_IDS})
