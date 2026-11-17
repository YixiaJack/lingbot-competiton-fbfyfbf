"""Build batch 2218 = cfa641b0 2218a27f (02:56), eps 33-35, all task 3, checked by eye 11/16 03:0x (b2 of 33/34 checked at full size: lift starts right at the boundary):
  own_t3_sub_new2218     demo 35 (4 subtasks) + follow-up pieces of DAgger 33/34 -> normal demo weight
  own_t3_dagger_2218_to  takeover pieces: 33 grasp (human from 589), 34 grasp (480) -> DAgger weight
Policy frames before each takeover are not used."""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import derive_dataset as DD
V1 = Path.home() / 'Downloads/cfa641b0_2218a27f/v1'; O = Path('out/ds_own2218'); O.mkdir(parents=True, exist_ok=True)
P = json.load(open('out/t3_2218.json'))
it = lambda vs: [{'src_ep': v['ep'], 'start': v['start'], 'end': v['end'], 'task': v['task'], 'subtask': v['subtask'], 'pan_arm': v['pan_arm'], 'kind': v['kind']}
                 for v in sorted(vs, key=lambda v: (v['ep'], v['start']))]
sub = [v for v in P.values() if v['kind'] in ('demo', 'follow')]; to = [v for v in P.values() if v['kind'] == 'takeover']
for name, vs in (('own_t3_sub_new2218', sub), ('own_t3_dagger_2218_to', to)):
    print(name, DD.build(V1, O / name, it(vs), 30), dict(collections.Counter(v['subtask'] for v in vs)))
json.dump({'note': 'cfa641b0 2218a27f task 3; checked by eye 2026-11-16 03:0x', 'pieces': P}, open(O / 't3_2218_final.json', 'w'), indent=1, ensure_ascii=False)
