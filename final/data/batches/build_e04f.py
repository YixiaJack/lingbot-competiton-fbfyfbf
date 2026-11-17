"""Build batch 25 = b60f1e2e e04f184c (00:15), eps 544-578, all normal teleop (classify_e04f.py: no policy frames):
  own_t12_newe04f          task-1 demos 548-557 + task-2 demos 558-578 (31), official-template prompts (instr_e04f.json)
  own_t12_box_v13_newe04f  30 of them with YOLO v13 box cues (out/owne04f_cues.json, all checked by eye 11/16 00:5x);
                           553 has no cue (v13 misses the green chili at frame 0) -> plain set only
  own_t3_sub_newe04f       task-3 demos 544-547, 4 subtask pieces each (out/t3_e04f.json, boundaries checked by eye)
Labelled partial 556 / 569 / 571 kept: potato on the purple plate, brush lying inside the middle compartment.
No DAgger in this batch.
Box dataset: derive_box_dataset.py afterwards (command at the bottom)."""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import derive_dataset as DD
V1 = Path.home() / 'Downloads/b60f1e2e_e04f184c/v1'; O = Path('out/ds_owne04f'); O.mkdir(parents=True, exist_ok=True)
fps = json.loads((V1 / 'meta/info.json').read_text())['fps']
# task 1
instr = {int(k): v for k, v in json.load(open('instr_e04f.json')).items()}
_, EPS, _ = DD.load_src(V1)
items = [{'src_ep': e, 'start': 0, 'end': EPS[e]['length'], 'task': instr[e]} for e in sorted(instr)]
print('own_t12_newe04f', DD.build(V1, O / 'own_t12_newe04f', items, fps))
c = json.load(open('out/owne04f_cues.json'))
c['summary']['checked_by_eye'] = '11/16 00:5x: all 30 cues correct (target, destination, grasp of the boxed object, final plate / compartment); 553 no cue (green chili not detected)'
assert len(c['episodes']) == 30 and 'ep0553' not in c['episodes'], sorted(c['episodes'])
json.dump(c, open('out/owne04f_cues.json', 'w'), indent=1, ensure_ascii=False)
# task 3
P = json.load(open('out/t3_e04f.json'))
ADJ = {}
for k, v in P.items():
    a = ADJ.get((v['ep'], v['subtask']))
    if a:
        print('adjust', k, a[0], v[a[0]], '->', a[1]); v[a[0]] = a[1]; v['secs'] = round((v['end'] - v['start']) / fps, 1); v['note'] = 'boundary moved by eye (dustpan lifts from 1028)'
assert all(v['kind'] == 'demo' for v in P.values())
it = lambda vs: [{'src_ep': v['ep'], 'start': v['start'], 'end': v['end'], 'task': v['task'], 'subtask': v['subtask'], 'pan_arm': v['pan_arm'], 'kind': v['kind']}
                 for v in sorted(vs, key=lambda v: (v['ep'], v['start']))]
print('own_t3_sub_newe04f', DD.build(V1, O / 'own_t3_sub_newe04f', it(P.values()), fps), dict(collections.Counter(v['subtask'] for v in P.values())))
json.dump({'note': 'batch 25 task 3; all 4 normal demos checked by eye 2026-11-16 00:5x', 'pieces': P},
          open(O / 't3_e04f_final.json', 'w'), indent=1, ensure_ascii=False)
# then:
# ~/lingbot_v7env/bin/python ~/lingbot-final-notes/experiments/autolabel/derive_box_dataset.py --src ~/Downloads/b60f1e2e_e04f184c/v1 \
#   --videos ~/Downloads/b60f1e2e_e04f184c/v1/videos --boxes out/owne04f_cues.json --out out/ds_owne04f/own_t12_box_v13_newe04f \
#   --instructions instr_e04f_dbd.json --workers 6
