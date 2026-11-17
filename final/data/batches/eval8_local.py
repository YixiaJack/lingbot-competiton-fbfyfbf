"""v7a vs v8 on the yolo8 hold-out (16 of the 49 new task-1 episodes 215-263, never trained on by v8), frame 0 of camera_top,
online cue rule. Truth = by-eye labels in yolo8/ds/final_labels.json. Sheet with the v8 cue drawn -> out/eval8_*.jpg"""
import json, sys
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '.')
import yolocue_bridge as YC, v7_boxes as VB
V1 = Path.home() / 'Downloads/b60f1e2e_0d3bf9f6/v1'
meta = {m['episode_index']: m for m in pq.read_table(sorted((V1 / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()}
FL = json.load(open('yolo8/ds/final_labels.json')); LAB = FL['labels']; HO = FL['holdout']
ins = {}
for f in ('instr_all.json', 'instr_0200.json'):
    for k, v in json.load(open(f)).items(): ins[int(k)] = v if isinstance(v, str) else v['instruction']
D = {n: YC.detector({**YC.DEFAULTS, 'weights': f'weights/{n}.torchscript', 'names': f'weights/{n}.names.json', 'device': 'cpu'}) for n in ('v7a', 'v8')}
area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
same = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1])) / max(min(area(a), area(b)), 1) > 0.5
F = ImageFont.load_default(size=13)
for split, eps in (('holdout', HO), ('trained', [int(k) for k in LAB if int(k) not in HO])):
    res = {n: {'cue': 0, 'right': 0, 'wrong': 0, 'why': {}} for n in D}; cells = []
    for e in sorted(eps):
        pr = YC.parse(ins.get(e, ''))
        if pr is None: print('no parse', e, ins.get(e)); continue
        task, tcls, dcls = pr; L = {}; [L.setdefault(k, b) for k, b in LAB[str(e)]]
        rgb = VB.frame0(V1, meta[e]); line = []
        for n, det in D.items():
            tb, db, why = YC.select(det(rgb), task, tcls, dcls, rgb=rgb); r = res[n]; r['why'][why] = r['why'].get(why, 0) + 1
            if tb:
                r['cue'] += 1; ok = tcls in L and same(tb, L[tcls]) and (dcls not in L or same(db, L[dcls]))
                r['right' if ok else 'wrong'] += 1; line.append(f"{n}:{'OK' if ok else 'WRONG'}")
            else: line.append(f'{n}:{why}')
            if n == 'v8':
                im = Image.fromarray(YC.draw(rgb, tb, db, False)) if tb else Image.fromarray(rgb)
        if split == 'holdout':
            c = Image.new('RGB', (480, 380), 'white'); c.paste(im.resize((480, 360)), (0, 20))
            ImageDraw.Draw(c).text((3, 3), f"ep{e} {tcls}->{dcls} | {' '.join(line)}", font=F, fill='black'); cells.append(c)
    for n, r in res.items(): print(split, n, json.dumps(r))
    if cells:
        sh = Image.new('RGB', (1920, ((len(cells) + 3) // 4) * 380), 'white')
        for i, c in enumerate(cells): sh.paste(c, ((i % 4) * 480, (i // 4) * 380))
        sh.save('out/eval8_holdout.jpg', quality=82)
