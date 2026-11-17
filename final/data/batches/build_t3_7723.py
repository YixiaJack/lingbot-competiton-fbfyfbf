"""Build batch 21 (b60f1e2e 77235417, eps 475-485, task 3) from out/t3_7723.json (all 11 episodes checked by eye: subtask
boundaries against the frames; every episode ends with the tools down and the balls in the basket).
  own_t3_sub_new7723     normal demos 478-485 (4 subtasks each) + follow-up pieces of DAgger 475-477 (sweep/dump/home
                         the human completed after taking over)  -> normal demo weight
  own_t3_dagger_7723_to  takeover pieces of 475-477 (all three taken over during grasp)                -> DAgger weight
Instructions = real_split.instruction; source_map carries subtask / pan_arm / kind."""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import derive_dataset as DD
V1 = Path.home() / 'Downloads/b60f1e2e_77235417/v1'; O = Path('out/ds_own7723'); O.mkdir(parents=True, exist_ok=True)
P = json.load(open('out/t3_7723.json'))
it = lambda vs: [{'src_ep': v['ep'], 'start': v['start'], 'end': v['end'], 'task': v['task'], 'subtask': v['subtask'], 'pan_arm': v['pan_arm'], 'kind': v['kind']}
                 for v in sorted(vs, key=lambda v: (v['ep'], v['start']))]
sub = [v for v in P.values() if v['kind'] in ('demo', 'follow')]; to = [v for v in P.values() if v['kind'] == 'takeover']
for name, vs in (('own_t3_sub_new7723', sub), ('own_t3_dagger_7723_to', to)):
    print(name, DD.build(V1, O / name, it(vs), 30), dict(collections.Counter(v['subtask'] for v in vs)))
json.dump({'note': 'batch 21 task 3; checked by eye 2026-11-15 17:00', 'pieces': P}, open(O / 't3_7723_final.json', 'w'), indent=1, ensure_ascii=False)
