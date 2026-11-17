"""False positives on held-out task-3 tables (no object of the 54 classes present): images with any detection >= 0.5,
and the ones that would make a task-1 cue (a plate or fruit >= 0.5). usage: python -I eval_neg.py MODEL [MODEL ...]"""
import json, sys, collections
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, '.')
import yolocue_bridge as YC
N = Path('yolo10/neg'); hold = set(json.load(open(N / 'holdout.json'))['holdout'])
imgs = [p for p in sorted((N / 'frames').glob('*.jpg')) if int(p.stem.split('_')[1][2:]) in hold]
live = sorted(Path('yolo10/live/yolocue').glob('*_nocue_raw.jpg'))
for n in sys.argv[1:]:
    det = YC.detector({**YC.DEFAULTS, 'weights': f'weights/{n}.torchscript', 'names': f'weights/{n}.names.json', 'device': 'cpu'})
    c = collections.Counter(); bad = 0
    for p in imgs:
        d = [x for x in det(np.asarray(Image.open(p).convert('RGB'))) if x[5] >= 0.5]; bad += bool(d)
        for x in d: c[x[0]] += 1
    lv = [[(x[0], round(x[5], 2)) for x in det(np.asarray(Image.open(p).convert('RGB'))) if x[5] >= 0.5] for p in live]
    print(n, f'task-3 hold-out images with a detection >= 0.5: {bad}/{len(imgs)}', dict(c.most_common(6)), '| live task-3 frames:', lv)
