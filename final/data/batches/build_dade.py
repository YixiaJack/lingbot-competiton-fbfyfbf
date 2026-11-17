"""Build batch dade = cfa641b0 dade648f (02:39), eps 30-32, all task 3 normal demos, checked by eye 11/16 02:4x:
  own_t3_sub_newdade     demos 30-32 (4 subtasks each) -> normal demo weight"""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import derive_dataset as DD
V1 = Path.home() / 'Downloads/cfa641b0_dade648f/v1'; O = Path('out/ds_owndade'); O.mkdir(parents=True, exist_ok=True)
P = json.load(open('out/t3_dade.json'))
it = lambda vs: [{'src_ep': v['ep'], 'start': v['start'], 'end': v['end'], 'task': v['task'], 'subtask': v['subtask'], 'pan_arm': v['pan_arm'], 'kind': v['kind']}
                 for v in sorted(vs, key=lambda v: (v['ep'], v['start']))]
sub = [v for v in P.values() if v['kind'] in ('demo', 'follow')]; to = [v for v in P.values() if v['kind'] == 'takeover']
for name, vs in (('own_t3_sub_newdade', sub),):
    print(name, DD.build(V1, O / name, it(vs), 30), dict(collections.Counter(v['subtask'] for v in vs)))
json.dump({'note': 'cfa641b0 dade648f task 3 (demos 30-32); checked by eye 2026-11-16 02:4x', 'pieces': P}, open(O / 't3_dade_final.json', 'w'), indent=1, ensure_ascii=False)
