"""Build batch 4f62 = cfa641b0 4f62f428 (03:17), eps 40-41, task 3 DAgger, checked by eye 11/16 03:2x.
User (03:2x): ep41 = the session that started at 03:09:43 (5080 session 31fcce4d, 03:09:44-03:11:13): drop its dump and
everything after it (dump 1792-2142, home 2142-2589). ep40 = session 692c8712 (03:07:16-03:08:54), kept whole.
  own_t3_sub_new4f62     follow-up pieces: ep40 dump + home, ep41 sweep -> normal demo weight
  own_t3_dagger_4f62_to  takeover pieces: ep40 sweep (human from 1827), ep41 grasp (671) -> DAgger weight"""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import derive_dataset as DD
V1 = Path.home() / 'Downloads/cfa641b0_4f62f428/v1'; O = Path('out/ds_own4f62'); O.mkdir(parents=True, exist_ok=True)
P = json.load(open('out/t3_4f62.json'))
DROP = {k for k, v in P.items() if v['ep'] == 41 and v['subtask'] in ('dump', 'home')}
print('dropped (user):', sorted(DROP)); assert len(DROP) == 2
P = {k: v for k, v in P.items() if k not in DROP}
it = lambda vs: [{'src_ep': v['ep'], 'start': v['start'], 'end': v['end'], 'task': v['task'], 'subtask': v['subtask'], 'pan_arm': v['pan_arm'], 'kind': v['kind']}
                 for v in sorted(vs, key=lambda v: (v['ep'], v['start']))]
sub = [v for v in P.values() if v['kind'] in ('demo', 'follow')]; to = [v for v in P.values() if v['kind'] == 'takeover']
for name, vs in (('own_t3_sub_new4f62', sub), ('own_t3_dagger_4f62_to', to)):
    print(name, DD.build(V1, O / name, it(vs), 30), sorted((v['ep'], v['subtask']) for v in vs))
json.dump({'note': 'cfa641b0 4f62f428 task 3; checked by eye 2026-11-16 03:2x; ep41 dump+home dropped at the user request (session 03:09:43)',
           'dropped': sorted(DROP), 'pieces': P}, open(O / 't3_4f62_final.json', 'w'), indent=1, ensure_ascii=False)
