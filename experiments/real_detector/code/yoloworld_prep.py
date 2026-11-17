"""Third-party open-source detector baseline: YOLO-World v2 (public weights) with the competition vocabulary encoded
once by CLIP and saved, so the saved model is a closed-set detector that needs no language model at inference.
The three organizer levels have no names a text prompt could tell apart: one 'organizer' class, split into levels by
position at selection time (see select_eval.py). Run on the login node (CPU). Usage: yoloworld_prep.py --root R
"""
import argparse, json
from pathlib import Path
from ultralytics import YOLOWorld
from catalog import CLASSES, NAME

PROMPT = {**{k: NAME[k] for k in CLASSES if not k.startswith(('plate_', 'organizer_'))},
          **{f'plate_{c}': f'{c} plate' for c in ('white', 'yellow', 'pink', 'purple', 'blue')},
          'brush': 'hand brush', 'basket': 'round woven basket', 'paper_ball_red': 'red crumpled paper ball',
          'paper_ball_white': 'white crumpled paper ball', 'lip_gloss': 'beige lip gloss tube',
          'highlighter': 'light round makeup compact', 'contour': 'dark round makeup compact'}
p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--size', default='l')
a = p.parse_args()
out = a.root / 'det' / 'yoloworld'; out.mkdir(parents=True, exist_ok=True)
keys = [k for k in CLASSES if not k.startswith('organizer_')] + ['organizer']
prompts = [PROMPT.get(k, 'clear acrylic makeup organizer' if k == 'organizer' else k) for k in keys]
m = YOLOWorld(str(a.root / 'weights' / f'yolov8{a.size}-worldv2.pt'))
m.set_classes(prompts)
m.save(str(out / f'yolov8{a.size}-worldv2-competition.pt'))
(out / 'classes.json').write_text(json.dumps({'keys': keys, 'prompts': prompts}, indent=1))
print(len(keys), 'classes saved')
