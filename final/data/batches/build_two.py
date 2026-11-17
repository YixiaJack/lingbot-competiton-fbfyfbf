"""Build batch 42da = b60f1e2e 42da9074 (01:35) eps 613-616 (task 1) and batch 2ffa = cfa641b0 2ffa8d81 (01:53) eps 20-23 (task 3),
all checked by eye 11/16 02:0x:
  own_t12_new42da          4 task-1 demos (official-template prompts; 616 keeps 'mandarin orange' (user 11/16 02:05; tangerine class))
  own_t12_box_v13_new42da  3 of them with YOLO v13 cues (616 tangerine not detected -> plain only); 613 partial kept (cucumber on the pink plate)
  own_t3_sub_new2ffa       demo 21 (4 subtasks) + follow-up pieces of DAgger 20/22/23 -> normal demo weight
  own_t3_dagger_2ffa_to    takeover pieces: 20 sweep (human from 1013), 22 grasp (485), 23 grasp (300) -> DAgger weight
Policy frames before each takeover are not used."""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import derive_dataset as DD
V1 = Path.home() / 'Downloads/b60f1e2e_42da9074/v1'; O = Path('out/ds_two'); O.mkdir(parents=True, exist_ok=True)
instr = {int(k): v for k, v in json.load(open('instr_42da.json')).items()}
_, EPS, _ = DD.load_src(V1)
print('own_t12_new42da', DD.build(V1, O / 'own_t12_new42da', [{'src_ep': e, 'start': 0, 'end': EPS[e]['length'], 'task': instr[e]} for e in sorted(instr)], 30))
c = json.load(open('out/own42da_cues.json')); c['summary']['checked_by_eye'] = '11/16 02:0x: 3 cues correct; 616 tangerine not detected (plain only)'
assert len(c['episodes']) == 3 and 'ep0616' not in c['episodes']; json.dump(c, open('out/own42da_cues.json', 'w'), indent=1, ensure_ascii=False)
V2 = Path.home() / 'Downloads/cfa641b0_2ffa8d81/v1'; P = json.load(open('out/t3_2ffa.json'))
it = lambda vs: [{'src_ep': v['ep'], 'start': v['start'], 'end': v['end'], 'task': v['task'], 'subtask': v['subtask'], 'pan_arm': v['pan_arm'], 'kind': v['kind']}
                 for v in sorted(vs, key=lambda v: (v['ep'], v['start']))]
sub = [v for v in P.values() if v['kind'] in ('demo', 'follow')]; to = [v for v in P.values() if v['kind'] == 'takeover']
for name, vs in (('own_t3_sub_new2ffa', sub), ('own_t3_dagger_2ffa_to', to)):
    print(name, DD.build(V2, O / name, it(vs), 30), dict(collections.Counter(v['subtask'] for v in vs)))
json.dump({'note': 'cfa641b0 2ffa8d81 task 3; checked by eye 2026-11-16 02:0x', 'pieces': P}, open(O / 't3_2ffa_final.json', 'w'), indent=1, ensure_ascii=False)
