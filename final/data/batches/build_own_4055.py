"""Build the b60f1e2e 4055af2a (01:24, batch 26) task-1/2 datasets, same recipe as e94f:
  own_t12_new4055          34 demos 579-612 (task 2: 579-601, task 1: 602-612; all normal teleop, classify_4055.py),
                           official-template prompts (instr_4055.json)
  own_t12_box_v13_new4055  31 of them with YOLO v13 box cues (out/own4055_cues.json, all checked by eye 11/16 01:5x);
                           no cue: 581/582 (small brush, the gripper command never opens past the threshold -> no grasp
                           frame; both demos are correct by eye) and 611 (lemon not detected) -> plain set only
All 34 labelled success; every one checked by eye (right object, right plate / compartment).
Box dataset: derive_box_dataset.py afterwards (command at the bottom)."""
import json, sys
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import derive_dataset as DD
V1 = Path.home() / 'Downloads/b60f1e2e_4055af2a/v1'; O = Path('out/ds_own4055'); O.mkdir(parents=True, exist_ok=True)
fps = json.loads((V1 / 'meta/info.json').read_text())['fps']
instr = {int(k): v for k, v in json.load(open('instr_4055.json')).items()}
START = {}
_, EPS, _ = DD.load_src(V1)
items = [{'src_ep': e, 'start': START.get(e, 0), 'end': EPS[e]['length'], 'task': instr[e]} for e in sorted(instr)]
print('own_t12_new4055', DD.build(V1, O / 'own_t12_new4055', items, fps))
c = json.load(open('out/own4055_cues.json'))
c['summary']['checked_by_eye'] = '11/16 01:5x: all 31 cues correct; 581/582 no grasp frame, 611 lemon not detected (plain only)'

assert len(c['episodes']) == 31 and not {'ep0581', 'ep0582', 'ep0611'} & set(c['episodes']), sorted(c['episodes'])
json.dump(c, open('out/own4055_cues.json', 'w'), indent=1, ensure_ascii=False)
print('cues', len(c['episodes']))
# then:
# ~/lingbot_v7env/bin/python ~/lingbot-final-notes/experiments/autolabel/derive_box_dataset.py --src ~/Downloads/b60f1e2e_4055af2a/v1 \
#   --videos ~/Downloads/b60f1e2e_4055af2a/v1/videos --boxes out/own4055_cues.json --out out/ds_own4055/own_t12_box_v13_new4055 \
#   --instructions instr_4055_dbd.json --workers 6
