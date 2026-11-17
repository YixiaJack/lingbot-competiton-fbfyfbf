"""ds_v10_off: the official train images of the v9 list with by-eye-checked corrections.
  - images with an object label the size of the plate under it (auto-label bug after placement), or a wrong/huge box, are dropped
  - palette pans labelled lipstick/contour are deleted (label only)
  - unlabelled objects found by v9 (>= 0.5) are added where the crop was checked by eye (rejects / relabels below)"""
import json, collections
from pathlib import Path
R = Path('off/home/guest/locany/yolo/ds_v6'); names = json.load(open('../weights/v9.names.json'))['names']; nid = {n: i for i, n in enumerate(names)}
S = json.load(open('off_suspects.json')); G = json.load(open('off_adds.json'))
REJECT = {117, 60, 70, 95, 51, 78, 144, 143, 47, 73, 149, 155, 176, 177, 148, 128, 178, 180, 183}
RELABEL = {69: 'papaya', 25: 'peach', 16: 'watermelon'}
drop = {s['img'] for s in S if s['why'] == 'plate-sized'} | {S[k]['img'] for k in (54, 290, 291, 292, 293, 294, 295, 296, 393, 423, 464, 481, 496, 569)}
dele = {(S[k]['img'], S[k]['i']) for k in range(571, 579)}
add = collections.defaultdict(list)
for g in G:
    if g['gid'] in REJECT: continue
    for m in g['members']: add[m['img']].append((RELABEL.get(g['gid'], g['cls']), m['box']))
O = Path('ds_off/labels/train'); O.mkdir(parents=True, exist_ok=True); st = collections.Counter(); keep = []
for p in sorted((R / 'images/train').glob('*.jpg')):
    if p.name in drop: st['dropped'] += 1; continue
    W, H = 640, 480
    lines = [l for l in (R / 'labels/train' / (p.stem + '.txt')).read_text().split('\n') if l.strip()]
    out = [l for i, l in enumerate(lines) if (p.name, i) not in dele]; st['deleted'] += len(lines) - len(out)
    for c, b in add.get(p.name, []):
        out.append(f"{nid[c]} {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}"); st['added'] += 1
    (O / (p.stem + '.txt')).write_text('\n'.join(out) + '\n'); keep.append(p.name)
(Path('ds_off') / 'keep.txt').write_text('\n'.join(keep) + '\n'); print(dict(st), 'kept', len(keep))
