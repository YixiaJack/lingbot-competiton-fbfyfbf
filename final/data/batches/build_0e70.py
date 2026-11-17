"""Build batch 0e70 = cfa641b0 0e70207b (03:50), eps 44-48, task 3 DAgger, checked by eye 11/16 03:5x
(out/chk_t3_0e70_1.jpg: every boundary correct; ep44 labelled 部分成功 but by eye it ends with the ball in the basket and
both tools down on the table, kept like earlier partials; ep45 sweep is only 4.7 s because the ball lay next to the pan).
User asked for "4条" at 03:5x; ep48 was preheated at 03:50 in the same snapshot and is included (成功).
  own_t3_sub_new0e70     follow-up pieces: dump + home of 44-48, plus ep45 sweep -> normal demo weight
  own_t3_dagger_0e70_to  takeover pieces: 44 sweep (human from 1080), 45 grasp (582), 46 sweep (776), 47 sweep (956),
                         48 sweep (1040) -> DAgger weight"""
import json, sys
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import derive_dataset as DD
V1 = Path.home() / 'Downloads/cfa641b0_0e70207b/v1'; O = Path('out/ds_own0e70'); O.mkdir(parents=True, exist_ok=True)
P = json.load(open('out/t3_0e70.json'))
it = lambda vs: [{'src_ep': v['ep'], 'start': v['start'], 'end': v['end'], 'task': v['task'], 'subtask': v['subtask'], 'pan_arm': v['pan_arm'], 'kind': v['kind']}
                 for v in sorted(vs, key=lambda v: (v['ep'], v['start']))]
sub = [v for v in P.values() if v['kind'] in ('demo', 'follow')]; to = [v for v in P.values() if v['kind'] == 'takeover']
assert len(sub) == 11 and len(to) == 5, (len(sub), len(to))
for name, vs in (('own_t3_sub_new0e70', sub), ('own_t3_dagger_0e70_to', to)):
    print(name, DD.build(V1, O / name, it(vs), 30), sorted((v['ep'], v['subtask']) for v in vs))
json.dump({'note': 'cfa641b0 0e70207b task 3 (DAgger 44-48); checked by eye 2026-11-16 03:5x; nothing dropped; ep44 partial kept (ends complete by eye)', 'pieces': P},
          open(O / 't3_0e70_final.json', 'w'), indent=1, ensure_ascii=False)
