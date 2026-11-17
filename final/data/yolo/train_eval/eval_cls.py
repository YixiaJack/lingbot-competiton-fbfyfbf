"""Per-class YOLO check on fully labelled (by-eye) images: for every true object, found (same class >= 0.5 overlapping),
confused (another class >= 0.5 overlapping, no same class), missed (nothing >= 0.5); plus detections >= 0.5 on no labelled
object (false). usage: python -I eval_cls.py MODEL IMG_DIR LABEL_DIR OUT.json"""
import json, sys, collections
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path.home() / 'yolocue'))
import yolocue_bridge as YC
n, idir, ldir, out = sys.argv[1:5]
cfg = {**YC.DEFAULTS, 'weights': f'weights/{n}.torchscript', 'names': f'weights/{n}.names.json', 'device': 'cpu'}
det = YC.detector(cfg); names = json.loads((YC.ROOT / cfg['names']).read_text())['names']
area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
same = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1])) / max(min(area(a), area(b)), 1) > 0.5
st = collections.defaultdict(lambda: collections.Counter()); conf = collections.Counter(); false = collections.Counter(); low = collections.defaultdict(list); bad = []
for p in sorted(Path(idir).glob('*.jpg')):
    im = np.asarray(Image.open(p).convert('RGB')); H, W = im.shape[:2]; L = []
    for line in (Path(ldir) / (p.stem + '.txt')).read_text().split('\n'):
        if line.strip():
            c, x, y, w, h = line.split(); x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H; L.append((names[int(c)], [x - w / 2, y - h / 2, x + w / 2, y + h / 2]))
    d = det(im)
    for c, b in L:
        ov = [x for x in d if same(x[1:5], b)]; s = [x for x in ov if x[0] == c]
        best = max([x[5] for x in s], default=0); low[c].append(round(best, 2))
        if best >= 0.5: st[c]['found'] += 1
        else:
            o = [x for x in ov if x[0] != c and x[5] >= 0.5]
            if o: st[c]['confused'] += 1; conf[(c, max(o, key=lambda x: x[5])[0])] += 1; bad.append([str(p), c, b, 'as ' + max(o, key=lambda x: x[5])[0]])
            else: st[c]['missed'] += 1; bad.append([str(p), c, b, f'missed {best:.2f}'])
    for x in d:
        if x[5] >= 0.5 and not any(same(x[1:5], b) for c, b in L): false[x[0]] += 1
json.dump({'per_class': {c: dict(v) for c, v in st.items()}, 'confusions': [[a, b, k] for (a, b), k in conf.most_common()], 'false': dict(false), 'best_conf': low, 'bad': bad}, open(out, 'w'))
rows = sorted(st.items(), key=lambda kv: kv[1]['found'] / max(sum(kv[1].values()), 1))
print('class: found/total (confused, missed)  median best same-class conf')
for c, v in rows:
    t = sum(v.values()); print(f'  {c:18s} {v["found"]:3d}/{t:<3d} ({v["confused"]}, {v["missed"]})  med {np.median(low[c]):.2f}')
print('confusions (true -> predicted):', conf.most_common(15)); print('false detections:', false.most_common(10))
