"""Build batch 24 = b60f1e2e c89c4a87 (22:22), eps 529-543, all normal teleop (classify_c89c.py: no policy frames):
  own_t12_newc89c          task-1 demos 529-532, official-template prompts (instr_c89c.json)
  own_t12_box_v13_newc89c  the same 4 with YOLO v13 box cues (out/ownc89c_cues.json, checked by eye 11/15 22:5x)
  own_t3_sub_newc89c       task-3 demos 533-543, 4 subtask pieces each (out/t3_c89c.json, real_split boundaries checked by
                           eye against the frames); 539: sweep/dump boundary moved 1049 -> 1028 (the dustpan already lifts
                           from ~1028, the split was 0.7 s late). Labelled partial 536-539, 542 kept: by eye every one ends
                           with all balls in the basket and both tools down on the table.
No DAgger in this batch (no own_t3_dagger_c89c_to / own_t1_dagger_c89c).
Box dataset: derive_box_dataset.py afterwards (command at the bottom)."""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import derive_dataset as DD
V1 = Path.home() / 'Downloads/b60f1e2e_c89c4a87/v1'; O = Path('out/ds_ownc89c'); O.mkdir(parents=True, exist_ok=True)
fps = json.loads((V1 / 'meta/info.json').read_text())['fps']
# task 1
instr = {int(k): v for k, v in json.load(open('instr_c89c.json')).items()}
_, EPS, _ = DD.load_src(V1)
items = [{'src_ep': e, 'start': 0, 'end': EPS[e]['length'], 'task': instr[e]} for e in sorted(instr)]
print('own_t12_newc89c', DD.build(V1, O / 'own_t12_newc89c', items, fps))
c = json.load(open('out/ownc89c_cues.json'))
c['summary']['checked_by_eye'] = '11/15 22:5x: all 4 cues correct (target, destination plate, grasp of the boxed object, final plate)'
assert sorted(c['episodes']) == ['ep0529', 'ep0530', 'ep0531', 'ep0532'] or len(c['episodes']) == 4, sorted(c['episodes'])
json.dump(c, open('out/ownc89c_cues.json', 'w'), indent=1, ensure_ascii=False)
# task 3
P = json.load(open('out/t3_c89c.json'))
ADJ = {(539, 'sweep'): ('end', 1028), (539, 'dump'): ('start', 1028)}
for k, v in P.items():
    a = ADJ.get((v['ep'], v['subtask']))
    if a:
        print('adjust', k, a[0], v[a[0]], '->', a[1]); v[a[0]] = a[1]; v['secs'] = round((v['end'] - v['start']) / fps, 1); v['note'] = 'boundary moved by eye (dustpan lifts from 1028)'
assert all(v['kind'] == 'demo' for v in P.values())
it = lambda vs: [{'src_ep': v['ep'], 'start': v['start'], 'end': v['end'], 'task': v['task'], 'subtask': v['subtask'], 'pan_arm': v['pan_arm'], 'kind': v['kind']}
                 for v in sorted(vs, key=lambda v: (v['ep'], v['start']))]
print('own_t3_sub_newc89c', DD.build(V1, O / 'own_t3_sub_newc89c', it(P.values()), fps), dict(collections.Counter(v['subtask'] for v in P.values())))
json.dump({'note': 'batch 24 task 3; all 11 normal demos checked by eye 2026-11-15 22:5x; 539 sweep/dump boundary moved to 1028', 'pieces': P},
          open(O / 't3_c89c_final.json', 'w'), indent=1, ensure_ascii=False)
# then:
# ~/lingbot_v7env/bin/python ~/lingbot-final-notes/experiments/autolabel/derive_box_dataset.py --src ~/Downloads/b60f1e2e_c89c4a87/v1 \
#   --videos ~/Downloads/b60f1e2e_c89c4a87/v1/videos --boxes out/ownc89c_cues.json --out out/ds_ownc89c/own_t12_box_v13_newc89c \
#   --instructions instr_c89c_dbd.json --workers 6
