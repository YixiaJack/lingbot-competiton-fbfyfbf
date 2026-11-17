"""Build batch 7928 = cfa641b0 7928a53e (03:28), eps 42-43, task 3 DAgger, both 成功, checked by eye 11/16 03:3x
(out/chk_t3_7928_1.jpg: sweep -> dump -> home boundaries correct, dustpan over the basket in dump, tools down in home).
  own_t3_sub_new7928     follow-up pieces: ep42 dump + home, ep43 dump + home -> normal demo weight
  own_t3_dagger_7928_to  takeover pieces: ep42 sweep (human from 852), ep43 sweep (human from 1633) -> DAgger weight"""
import json, sys
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import derive_dataset as DD
V1 = Path.home() / 'Downloads/cfa641b0_7928a53e/v1'; O = Path('out/ds_own7928'); O.mkdir(parents=True, exist_ok=True)
P = json.load(open('out/t3_7928.json'))
it = lambda vs: [{'src_ep': v['ep'], 'start': v['start'], 'end': v['end'], 'task': v['task'], 'subtask': v['subtask'], 'pan_arm': v['pan_arm'], 'kind': v['kind']}
                 for v in sorted(vs, key=lambda v: (v['ep'], v['start']))]
sub = [v for v in P.values() if v['kind'] in ('demo', 'follow')]; to = [v for v in P.values() if v['kind'] == 'takeover']
assert len(sub) == 4 and len(to) == 2, (len(sub), len(to))
for name, vs in (('own_t3_sub_new7928', sub), ('own_t3_dagger_7928_to', to)):
    print(name, DD.build(V1, O / name, it(vs), 30), sorted((v['ep'], v['subtask']) for v in vs))
json.dump({'note': 'cfa641b0 7928a53e task 3; checked by eye 2026-11-16 03:3x; nothing dropped', 'pieces': P},
          open(O / 't3_7928_final.json', 'w'), indent=1, ensure_ascii=False)
