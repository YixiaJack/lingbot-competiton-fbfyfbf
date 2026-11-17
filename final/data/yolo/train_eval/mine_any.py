"""Find suspect labels in the official (ds_v6, auto-labelled, never checked by eye) images of the YOLO train list by
comparing each label with YOLO v9 on the same image (inference env, CPU).
  cls    label box where v9 sees another class (>= 0.5) and not the labelled class (>= 0.25)
  ghost  label box where v9 sees nothing (any class >= 0.1)
  miss   v9 box >= 0.7 that no label overlaps (an object left unlabelled = taught as background)
usage: python -I mine_v6.py TRAIN_LIST OUT.json DIR_SUBSTR[,DIR_SUBSTR]"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path.home() / 'yolocue'))
import yolocue_bridge as YC
lst, out = sys.argv[1], sys.argv[2]
cfg = {**YC.DEFAULTS, 'weights': 'weights/v9.torchscript', 'names': 'weights/v9.names.json', 'device': 'cpu'}
det = YC.detector(cfg); names = json.loads((YC.ROOT / cfg['names']).read_text())['names']
imgs = sorted({l.strip() for l in open(lst) if any(s in l for s in sys.argv[3].split(','))})
area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
inter = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
iou = lambda a, b: inter(a, b) / (area(a) + area(b) - inter(a, b) + 1e-9)
res = {}
for k, p in enumerate(imgs):
    im = np.asarray(Image.open(p).convert('RGB')); H, W = im.shape[:2]
    lp = Path(p.replace('/images/', '/labels/')).with_suffix('.txt'); L = []
    for line in (lp.read_text().split('\n') if lp.exists() else []):
        if not line.strip(): continue
        c, x, y, w, h = line.split(); x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
        L.append((names[int(c)], [x - w / 2, y - h / 2, x + w / 2, y + h / 2]))
    d = det(im); bad = []
    for c, b in L:
        ov = [x for x in d if iou(x[1:5], b) > 0.5]
        same = any(x[0] == c and x[5] >= 0.25 for x in ov); other = [x for x in ov if x[0] != c and x[5] >= 0.5]
        if not same and other: bad.append({'t': 'cls', 'label': c, 'v9': other[0][0], 'conf': round(other[0][5], 2), 'box': [round(v) for v in b]})
        elif not any(x[5] >= 0.1 for x in ov): bad.append({'t': 'ghost', 'label': c, 'box': [round(v) for v in b]})
        elif not c.startswith(('plate_', 'organizer_')) and any(cc.startswith('plate_') and iou(b, pb) > 0.6 for cc, pb in L): bad.append({'t': 'plate-sized', 'label': c, 'box': [round(v) for v in b]})
    for x in d:
        if x[5] >= 0.7 and not any(iou(x[1:5], b) > 0.3 for c, b in L): bad.append({'t': 'miss', 'v9': x[0], 'conf': round(x[5], 2), 'box': [round(v) for v in x[1:5]]})
    if bad: res[p] = {'labels': L, 'bad': bad}
    if k % 500 == 0: print(k, len(res), flush=True)
json.dump(res, open(out, 'w'))
import collections
print('images', len(imgs), 'with suspects', len(res), collections.Counter(b['t'] for v in res.values() for b in v['bad']))
print('cls pairs', collections.Counter((b['label'], b['v9']) for v in res.values() for b in v['bad'] if b['t'] == 'cls').most_common(25))
print('miss', collections.Counter(b['v9'] for v in res.values() for b in v['bad'] if b['t'] == 'miss').most_common(15))
print('ghost', collections.Counter(b['label'] for v in res.values() for b in v['bad'] if b['t'] == 'ghost').most_common(15))
