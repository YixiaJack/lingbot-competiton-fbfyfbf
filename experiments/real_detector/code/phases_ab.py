"""Append the cue-A/B phases to phases.txt (after the detector phases). Gates: EXPERT_OK (I create it once the expert
works), COLLECTION_READY / gates/<run>[_YOLO] (controller_ab.py). Eval items are batches of 10 (chunks 0-5)."""
import os
from pathlib import Path
R = Path(os.path.expandvars('$SCRATCH/lingbot/final_real_20261112')); C = R / 'code'
DEPS = os.path.expandvars('PYTHONPATH=$SCRATCH/lingbot/final_yolo_20261111/deps')
(R / 'lists/chunks.txt').write_text(''.join(f'{i}\n' for i in range(6)))
(R / 'lists/chunks_half.txt').write_text(''.join(f'{i}\n' for i in (0, 3)))   # 10 task-1 + 10 task-2 items
(R / 'lists/collect_t1.txt').write_text(''.join(f'{i}\n' for i in range(200)))
(R / 'lists/collect_t2.txt').write_text(''.join(f'{i}\n' for i in range(160, 260)))
(R / 'lists/eval_t1.txt').write_text(''.join(f'{i}\n' for i in range(60)))
(R / 'lists/eval_t2.txt').write_text(''.join(f'{i}\n' for i in range(30, 60)))
# task 1 and task 2 experts are released separately (EXPERT_OK / EXPERT_T2_OK): a broken expert must not burn GPU hours
L = [f'collect_t1|{R}/lists/collect_t1.txt|bash {C}/runtime.sh sim python {C}/collect_real.py --root {R} --item {{item}}|{R}/EXPERT_T1_OK',
     f'evalseed_t1|{R}/lists/eval_t1.txt|bash {C}/runtime.sh sim python {C}/eval_real.py --root {R} --run x --mode text --item {{item}} --seed-only|{R}/EXPERT_T1_OK',
     f'collect_t2|{R}/lists/collect_t2.txt|bash {C}/runtime.sh sim python {C}/collect_real.py --root {R} --item {{item}}|{R}/EXPERT_T2_OK',
     f'evalseed_t2|{R}/lists/eval_t2.txt|bash {C}/runtime.sh sim python {C}/eval_real.py --root {R} --run x --mode text --item {{item}} --seed-only|{R}/EXPERT_T2_OK',
     f'convert|{R}/lists/variants.txt|bash {C}/runtime.sh model python {C}/prepare_data_real.py --root {R} --variant {{item}}|{R}/COLLECTION_READY']
# 22k+8k only (11/12, the team's 2:18 choice): text vs box on 60 scenes; the decision-relevant extras on the box model
# at full size (no cue: does the shared weight still work without a box; yolo: the real pipeline), robustness on 20
ev = [('s22_text', 'text', 'oracle', 'chunks', 's22_text'), ('s22_box', 'box', 'oracle', 'chunks', 's22_box'),
      ('s22_box', 'text', 'oracle', 'chunks', 's22_box'), ('s22_box', 'box', 'yolo', 'chunks', 's22_box_YOLO'),
      ('s22_box', 'box', 'jitter', 'chunks_half', 's22_box'), ('s22_box', 'box', 'wrong10', 'chunks_half', 's22_box'),
      # 11/14: target box dropped after the grasp; the same three decision-relevant conditions on the same 60 scenes
      ('s22_boxgrasp', 'boxgrasp', 'oracle', 'chunks', 's22_boxgrasp'), ('s22_boxgrasp', 'text', 'oracle', 'chunks', 's22_boxgrasp'),
      ('s22_boxgrasp', 'boxgrasp', 'yolo', 'chunks', 's22_boxgrasp_YOLO')]
for run, mode, src, lst, gate in ev:
    pre = f'{DEPS} ' if src == 'yolo' else ''
    L.append(f'{run}_{mode}_{src}|{R}/lists/{lst}.txt|{pre}bash {C}/eval_batch.sh {run} {mode} {src} {{item}}|{R}/gates/{gate}')
# 16 GB check (the venue's RTX 5080): the box model with YOLO boxes, server memory capped at 15 GB, 10 episodes
(R / 'lists/chunk0.txt').write_text('0\n')
L.append(f'mem15|{R}/lists/chunk0.txt|{DEPS} bash {C}/eval_batch.sh s22_box box yolo {{item}} 15|{R}/gates/s22_box_YOLO')
# YOLO-World last (decided 11/12): only once our detector, the 6B plug-in check and every A/B episode are done
L.append(f'yoloworld|{R}/lists/single.txt|{DEPS} bash {C}/runtime.sh model python {C}/select_eval.py --root {R} '
         f'--weights {R}/det/yoloworld/yolov8l-worldv2-competition.pt --classes {R}/det/yoloworld/classes.json '
         f'--tag _yoloworld|{R}/gates/OURS_DONE')
p = R / 'phases.txt'
old = [l for l in p.read_text().splitlines() if l.split('|')[0] in ('det', 'yolo', 'select')]
p.write_text('\n'.join(old + L) + '\n')
print(len(old), 'detector phases kept,', len(L), 'A/B phases')
