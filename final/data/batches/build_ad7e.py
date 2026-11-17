"""Build batch ad7e = cfa641b0 ad7eee6c (02:32), eps 24-29, all task 3, checked by eye 11/16 02:4x:
  own_t3_sub_newad7e     demos 28/29 (4 subtasks each) + follow-up pieces of DAgger 24-27 -> normal demo weight
  own_t3_dagger_ad7e_to  takeover pieces: 24 grasp (human from 356), 25 sweep (802), 26 sweep (1004), 27 grasp (518) -> DAgger weight
Policy frames before each takeover are not used."""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import derive_dataset as DD
V1 = Path.home() / 'Downloads/cfa641b0_ad7eee6c/v1'; O = Path('out/ds_ownad7e'); O.mkdir(parents=True, exist_ok=True)
P = json.load(open('out/t3_ad7e.json'))
it = lambda vs: [{'src_ep': v['ep'], 'start': v['start'], 'end': v['end'], 'task': v['task'], 'subtask': v['subtask'], 'pan_arm': v['pan_arm'], 'kind': v['kind']}
                 for v in sorted(vs, key=lambda v: (v['ep'], v['start']))]
sub = [v for v in P.values() if v['kind'] in ('demo', 'follow')]; to = [v for v in P.values() if v['kind'] == 'takeover']
for name, vs in (('own_t3_sub_newad7e', sub), ('own_t3_dagger_ad7e_to', to)):
    print(name, DD.build(V1, O / name, it(vs), 30), dict(collections.Counter(v['subtask'] for v in vs)))
json.dump({'note': 'cfa641b0 ad7eee6c task 3; checked by eye 2026-11-16 02:4x', 'pieces': P}, open(O / 't3_ad7e_final.json', 'w'), indent=1, ensure_ascii=False)
