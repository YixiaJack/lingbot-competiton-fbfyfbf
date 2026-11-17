"""Official (ds_v6) train images: find unlabelled objects (v9 >= 0.5, no label overlapping) and suspicious existing labels.
Adds of the same object (same episode, class, centre within 40 px) form one group -> one crop to check by eye.
Writes off_adds.json, off_suspects.json, crop grids adds_<class>_N.jpg (group id under each crop)."""
import json, sys, collections
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '..')
import yolocue_bridge as YC
R = Path('off/home/guest/locany/yolo/ds_v6'); imgs = sorted((R / 'images/train').glob('*.jpg'))
det = YC.detector({**YC.DEFAULTS, 'weights': 'weights/v9.torchscript', 'names': 'weights/v9.names.json', 'device': 'cpu'})
names = json.load(open('../weights/v9.names.json'))['names']
area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
inter = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
iou = lambda a, b: inter(a, b) / (area(a) + area(b) - inter(a, b) + 1e-9)
ctr = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
adds, sus = [], []
for k, p in enumerate(imgs):
    im = np.asarray(Image.open(p).convert('RGB')); H, W = im.shape[:2]; L = []
    for line in (R / 'labels/train' / (p.stem + '.txt')).read_text().split('\n'):
        if line.strip():
            c, x, y, w, h = line.split(); x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H; L.append((names[int(c)], [x - w / 2, y - h / 2, x + w / 2, y + h / 2]))
    d = det(im); took = []
    for x in sorted(d, key=lambda x: -x[5]):
        if x[5] < 0.5: break
        b = x[1:5]
        if any(iou(b, lb) > 0.3 for c, lb in L) or any(iou(b, t) > 0.6 for t in took): continue
        if not x[0].startswith(('plate_', 'organizer_')) and any(c == x[0] and inter(b, lb) / max(area(b), 1) > 0.6 for c, lb in L): continue
        took.append(b); adds.append({'img': p.name, 'cls': x[0], 'conf': round(x[5], 3), 'box': [round(v, 1) for v in b]})
    for i, (c, lb) in enumerate(L):
        ov = [x for x in d if iou(x[1:5], lb) > 0.5]
        why = None
        if not any(x[0] == c and x[5] >= 0.25 for x in ov) and any(x[0] != c and x[5] >= 0.5 for x in ov): why = 'cls ' + max((x for x in ov if x[0] != c), key=lambda x: x[5])[0]
        elif not any(x[5] >= 0.1 for x in ov): why = 'ghost'
        elif not c.startswith(('plate_', 'organizer_')) and any(cc.startswith('plate_') and iou(lb, pb) > 0.6 for cc, pb in L): why = 'plate-sized'
        if why: sus.append({'img': p.name, 'i': i, 'cls': c, 'box': [round(v, 1) for v in lb], 'why': why})
    if k % 700 == 0: print(k, len(adds), len(sus), flush=True)
groups = []
for a in adds:
    ep = a['img'].split('_')[0]; c = ctr(a['box'])
    g = next((g for g in groups if g['ep'] == ep and g['cls'] == a['cls'] and abs(ctr(g['members'][0]['box'])[0] - c[0]) < 40 and abs(ctr(g['members'][0]['box'])[1] - c[1]) < 40), None)
    if g: g['members'].append(a)
    else: groups.append({'gid': len(groups), 'ep': ep, 'cls': a['cls'], 'members': [a]})
json.dump(groups, open('off_adds.json', 'w')); json.dump(sus, open('off_suspects.json', 'w'))
print('adds', len(adds), 'groups', len(groups), 'suspects', len(sus), collections.Counter(s['why'].split()[0] for s in sus))
print(collections.Counter(g['cls'] for g in groups).most_common())
F = ImageFont.load_default(size=11)
bycls = collections.defaultdict(list)
for g in groups: bycls[g['cls']].append(g)
order = sorted(bycls, key=lambda c: names.index(c)); allcells = []
for c in order:
    for g in bycls[c]:
        a = max(g['members'], key=lambda m: m['conf']); b = a['box']; im = Image.open(R / 'images/train' / a['img']).convert('RGB')
        m = 12; cr = im.crop((max(b[0] - m, 0), max(b[1] - m, 0), min(b[2] + m, im.width), min(b[3] + m, im.height))); cr.thumbnail((116, 100))
        cell = Image.new('RGB', (120, 118), 'white'); cell.paste(cr, ((120 - cr.width) // 2, 2)); dd = ImageDraw.Draw(cell)
        dd.text((2, 104), f"{g['gid']} {c[:11]} {a['conf']:.2f}", font=F, fill='black'); allcells.append((c, cell))
per = 96; sheets = 0
for k in range(0, len(allcells), per):
    cc = allcells[k:k + per]; sh = Image.new('RGB', (12 * 120, ((len(cc) + 11) // 12) * 118), (200, 200, 200))
    for i, (c, cell) in enumerate(cc): sh.paste(cell, ((i % 12) * 120, (i // 12) * 118))
    sheets += 1; sh.save(f'adds_{sheets:02d}.jpg', quality=85)
print('sheets', sheets)
