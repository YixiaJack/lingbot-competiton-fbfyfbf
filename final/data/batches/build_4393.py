"""Build batch 4393 = cfa641b0 4393fff3 (03:07), eps 36-39, all task 3 DAgger, checked by eye 11/16 03:1x (ep36 split with a per-arm 0.30 threshold for the shallow arm-1 release):
  own_t3_sub_new4393     follow-up pieces of DAgger 36-39 -> normal demo weight
  own_t3_dagger_4393_to  takeover pieces: 36 grasp (human from 252), 37 grasp (249), 38 sweep (753), 39 sweep (1077) -> DAgger weight
Policy frames before each takeover are not used."""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import derive_dataset as DD
V1 = Path.home() / 'Downloads/cfa641b0_4393fff3/v1'; O = Path('out/ds_own4393'); O.mkdir(parents=True, exist_ok=True)
P = json.load(open('out/t3_4393.json'))
it = lambda vs: [{'src_ep': v['ep'], 'start': v['start'], 'end': v['end'], 'task': v['task'], 'subtask': v['subtask'], 'pan_arm': v['pan_arm'], 'kind': v['kind']}
                 for v in sorted(vs, key=lambda v: (v['ep'], v['start']))]
sub = [v for v in P.values() if v['kind'] in ('demo', 'follow')]; to = [v for v in P.values() if v['kind'] == 'takeover']
for name, vs in (('own_t3_sub_new4393', sub), ('own_t3_dagger_4393_to', to)):
    print(name, DD.build(V1, O / name, it(vs), 30), dict(collections.Counter(v['subtask'] for v in vs)))
json.dump({'note': 'cfa641b0 4393fff3 task 3 (DAgger 36-39); checked by eye 2026-11-16 03:1x', 'pieces': P}, open(O / 't3_4393_final.json', 'w'), indent=1, ensure_ascii=False)
