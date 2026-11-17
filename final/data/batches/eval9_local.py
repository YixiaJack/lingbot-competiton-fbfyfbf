"""Compare YOLO versions on by-eye-labelled hold-out episodes, frame 0 of camera_top (what the service sees).
  cue: online rule (YC.select on the parsed instruction); right = white box on the true target and orange on the true destination
  obj: every labelled object; found = a same-class detection >= 0.5 overlapping it; extra = detections >= 0.5 matching no same-class label
usage: python -I eval9_local.py LABELS.json INSTR1.json[,INSTR2.json] SHEET_PREFIX MODEL [MODEL ...]   (MODEL = weights/<name>.torchscript)"""
import json, sys
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '.')
import yolocue_bridge as YC, v7_boxes as VB
V1 = Path.home() / 'Downloads/b60f1e2e_0d3bf9f6/v1'
lab_f, instr_fs, prefix, models = sys.argv[1], sys.argv[2].split(','), sys.argv[3], sys.argv[4:]
meta = {m['episode_index']: m for m in pq.read_table(sorted((V1 / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()}
FL = json.load(open(lab_f)); LAB = FL['labels']; HO = FL['holdout']
ins = {}
for f in instr_fs:
    for k, v in json.load(open(f)).items(): ins[int(k)] = v if isinstance(v, str) else v['instruction']
D = {n: YC.detector({**YC.DEFAULTS, 'weights': f'weights/{n}.torchscript', 'names': f'weights/{n}.names.json', 'device': 'cpu'}) for n in models}
area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
same = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1])) / max(min(area(a), area(b)), 1) > 0.5
F = ImageFont.load_default(size=13)
res = {n: {'cue': 0, 'right': 0, 'wrong': 0, 'why': {}, 'obj': 0, 'found': 0, 'extra': 0, 'extra_list': []} for n in D}; cells = {n: [] for n in D}
for e in sorted(HO):
    pr = YC.parse(ins.get(e, ''))
    task, tcls, dcls = pr; truth = LAB[str(e)]; L = {}
    for k, b in truth: L.setdefault(k, b)
    rgb = VB.frame0(V1, meta[e])
    for n, det in D.items():
        d = det(rgb); r = res[n]
        tb, db, why = YC.select(d, task, tcls, dcls, rgb=rgb); r['why'][why] = r['why'].get(why, 0) + 1
        if tb:
            r['cue'] += 1; ok = tcls in L and same(tb, L[tcls]) and (dcls not in L or same(db, L[dcls])); r['right' if ok else 'wrong'] += 1; tag = 'OK' if ok else 'WRONG'
        else: tag = why
        hi = [x for x in d if x[5] >= 0.5]
        for k, b in truth:
            r['obj'] += 1; r['found'] += any(x[0] == k and same(x[1:5], b) for x in hi)
        for x in hi:
            if not any(x[0] == k and same(x[1:5], b) for k, b in truth): r['extra'] += 1; r['extra_list'].append(f'ep{e}:{x[0]}:{x[5]:.2f}')
        im = Image.fromarray(YC.draw(rgb, tb, db, False)) if tb else Image.fromarray(rgb); g = ImageDraw.Draw(im)
        for x in hi: g.text((x[1] + 2, x[4] - 12), f'{x[0]} {x[5]:.2f}', font=F, fill=(255, 255, 0))
        c = Image.new('RGB', (480, 380), 'white'); c.paste(im.resize((480, 360)), (0, 20))
        ImageDraw.Draw(c).text((3, 3), f'{n} ep{e} {tcls}->{dcls} | {tag}', font=F, fill='black' if tag == 'OK' else (200, 0, 0)); cells[n].append(c)
for n, r in res.items():
    print(n, json.dumps({k: v for k, v in r.items() if k != 'extra_list'}), 'extra:', r['extra_list'][:12])
    cc = cells[n]; sh = Image.new('RGB', (1920, ((len(cc) + 3) // 4) * 380), 'white')
    for i, c in enumerate(cc): sh.paste(c, ((i % 4) * 480, (i // 4) * 380))
    sh.save(f'{prefix}_{n}.jpg', quality=80)
