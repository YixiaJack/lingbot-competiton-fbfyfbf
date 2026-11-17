"""Build the b60f1e2e e94f77bf (21:36, batch 23) task-2 datasets, same recipe as 908a:
  own_t12_newe94f          18 task-2 demos 511-528 (all normal teleop: classify_e94f.py found no policy frames),
                           official-template prompts (instr_e94f.json); 516 starts at frame 20 (an operator's hand is
                           still placing the contour compact in frames 0-~12)
  own_t12_box_v13_newe94f  17 of them with YOLO v13 box cues (out/owne94f_cues.json, all 17 checked by eye 11/15 22:2x);
                           516 has no cue (target not detected at frame 0, hand in view) -> plain set only
Labelled partial 519 / 527 / 528 kept: by eye the brush / palette ends inside the middle compartment (519 = several
grasp attempts on the thin brush, 527/528 = the palette is tilted to fit); the right object moved in every one.
Box dataset: derive_box_dataset.py afterwards (command at the bottom)."""
import json, sys
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import derive_dataset as DD
V1 = Path.home() / 'Downloads/b60f1e2e_e94f77bf/v1'; O = Path('out/ds_owne94f'); O.mkdir(parents=True, exist_ok=True)
fps = json.loads((V1 / 'meta/info.json').read_text())['fps']
instr = {int(k): v for k, v in json.load(open('instr_e94f.json')).items()}
START = {516: 20}
_, EPS, _ = DD.load_src(V1)
items = [{'src_ep': e, 'start': START.get(e, 0), 'end': EPS[e]['length'], 'task': instr[e]} for e in sorted(instr)]
print('own_t12_newe94f', DD.build(V1, O / 'own_t12_newe94f', items, fps))
c = json.load(open('out/owne94f_cues.json'))
c['summary']['checked_by_eye'] = ('11/15 22:2x: all 17 cues correct (target and destination boxes, grasp of the boxed object, '
                                  'final compartment); 516 no cue (hand in frame 0); partial 519/527/528 end in the right compartment')
assert '516' not in c['episodes'] and len(c['episodes']) == 17, sorted(c['episodes'])
json.dump(c, open('out/owne94f_cues.json', 'w'), indent=1, ensure_ascii=False)
print('cues', len(c['episodes']))
# then:
# ~/lingbot_v7env/bin/python ~/lingbot-final-notes/experiments/autolabel/derive_box_dataset.py --src ~/Downloads/b60f1e2e_e94f77bf/v1 \
#   --videos ~/Downloads/b60f1e2e_e94f77bf/v1/videos --boxes out/owne94f_cues.json --out out/ds_owne94f/own_t12_box_v13_newe94f \
#   --instructions instr_e94f_dbd.json --workers 6
