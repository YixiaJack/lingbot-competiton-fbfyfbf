"""YOLO v13 labels: own task-1 episodes with the v13 focus objects (pear, red apple / red delicious / pomegranate / tomato, potato as the pear look-alike). Per episode: frames before
the pick-up (0 and every 15 frames while < grasp_frame - 20, at most 5), proposals on frame 0 = v8 detections
(>= 0.25, cross-class NMS) + the target located by first/last frame difference (class from the instruction) + low
v8 boxes (0.05-0.25, class unknown) + colour blobs nothing covers. Review sheets number every proposal."""
import json, sys
from pathlib import Path
import av, cv2, numpy as np, pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont, ImageFilter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yolocue_bridge as YC, v7_boxes as VB
V1 = Path.home() / 'Downloads/b60f1e2e_77235417/v1'; OUT = Path(__file__).resolve().parent / 'prop'
VIDS = [Path.home() / f'Downloads/{d}/v1' for d in ('b60f1e2e_77235417', 'b60f1e2e_3541a365', 'b60f1e2e_0d3bf9f6')]; (OUT / 'frames').mkdir(parents=True, exist_ok=True)
EPS = [22, 414, 418, 107, 108, 233, 234, 8, 9, 109, 264, 19, 229, 231, 331, 31, 32, 112, 313, 316]
meta = {m['episode_index']: m for m in pq.read_table(sorted((V1 / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()}
import re as _re
lab = {}
for _l in open(V1 / 'meta/labels.jsonl'):
    _r = json.loads(_l); _L = (_r.get('labels') or [{}])[0]; lab[_r['episode_index']] = _L.get('task_instruction') or ''
clean = lambda s: _re.sub(r'\b(onion|mangosteen) fruit\b', r'\1', s).replace('the apple', 'the red apple')
instr = {e: clean(lab[e]) for e in EPS}
t = pq.read_table(sorted((V1 / 'data').glob('*/*.parquet'))[0], columns=['episode_index', 'frame_index', 'action.effector.position', 'observation.state.effector.position'])
ep, fi = t.column('episode_index').to_numpy(), t.column('frame_index').to_numpy(); C = np.array(t.column('action.effector.position').to_pylist(), float); G = np.array(t.column('observation.state.effector.position').to_pylist(), float)
det = YC.Detector(YC.ROOT / 'weights/v12.torchscript', json.loads((YC.ROOT / 'weights/v12.names.json').read_text())['names'], conf=0.05)
area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
inter = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
iou = lambda a, b: inter(a, b) / (area(a) + area(b) - inter(a, b) + 1e-9)
CAM = VB.CAM
def vpath(m):
    rel = Path('videos') / CAM / f"chunk-{m[f'videos/{CAM}/chunk_index']:03d}" / f"file-{m[f'videos/{CAM}/file_index']:03d}.mp4"
    return next(d / rel for d in VIDS if (d / rel).exists())
def frames(m, idxs):
    p = vpath(m); t0 = m[f'videos/{CAM}/from_timestamp']
    want = {round(t0 + i / 30, 4): i for i in idxs}; got = {}
    with av.open(str(p)) as ic:
        s = ic.streams.video[0]; ic.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s, backward=True)
        for fr in ic.decode(s):
            ft = float(fr.pts * s.time_base)
            for tw, i in want.items():
                if i not in got and ft + 1e-3 >= tw: got[i] = fr.to_ndarray(format='rgb24')
            if len(got) == len(want): break
    return got
def lastf(m):
    p = vpath(m); t1 = m[f'videos/{CAM}/to_timestamp'] - 0.1
    with av.open(str(p)) as ic:
        s = ic.streams.video[0]; ic.seek(int(max(t1 - 1, 0) / s.time_base), stream=s, backward=True); b = None
        for fr in ic.decode(s):
            if float(fr.pts * s.time_base) > t1: break
            b = fr
        return b.to_ndarray(format='rgb24')
def diff_start(a, b):
    A = np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(2)), np.float32); B = np.asarray(Image.fromarray(b).filter(ImageFilter.GaussianBlur(2)), np.float32)
    H = A.shape[0]; m = (np.abs(A - B).max(2) > 45).astype(np.uint8); m[:int(H * .25)] = 0
    k = np.ones((3, 3), np.uint8); m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2), cv2.MORPH_CLOSE, k, iterations=4)
    low = np.zeros_like(m, bool); low[H // 2:] = True; tab = np.median(A[low & (m == 0)], 0)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m); best = None
    for i in range(1, n):
        x, y, w, h, ar = [int(v) for v in st[i]]
        if ar < 150: continue
        sl = (slice(y, y + h), slice(x, x + w)); comp = lab[sl] == i
        g = float((np.linalg.norm(B[sl][comp] - tab, axis=1) < 35).mean()) - float((np.linalg.norm(A[sl][comp] - tab, axis=1) < 35).mean())
        if g > 0.2 and (best is None or g * np.sqrt(ar) > best[0]): best = (g * np.sqrt(ar), [x, y, x + w, y + h])
    return best[1] if best else None
def blobs(rgb, known):
    lab = cv2.cvtColor(cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32); H = lab.shape[0]
    tab = np.median(lab[int(H * .55):].reshape(-1, 3), 0); table = (np.linalg.norm(lab[..., 1:] - tab[1:], axis=2) < 11) & (np.abs(lab[..., 0] - tab[0]) < 55)
    frac = table.mean(1); top = next((y for y in range(H // 5, H - 20) if frac[y:y + 12].mean() > 0.55), int(H * .4))
    fg = (~table).astype(np.uint8); fg[:top + 6] = 0
    for b in known: fg[max(int(b[1]) - 4, 0):int(b[3]) + 4, max(int(b[0]) - 4, 0):int(b[2]) + 4] = 0
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    n, _, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=3)); out = []
    for i in range(1, n):
        x, y, w, h, a = [int(v) for v in st[i]]
        if 350 <= a <= 30000 and min(w, h) >= 14 and max(w, h) / min(w, h) < 7 and a / (w * h) > 0.25: out.append([x, y, x + w, y + h])
    return out
def blocked(c, g):
    """first frame after a gripper opened where the measured gripper stays >= 0.08 above the command for 10 frames"""
    best = None
    for x in (0, 1):
        if c[:, x].max() < 0.25: continue
        j0 = int(np.argmax(c[:, x] > 0.25)); run = 0
        for j in range(j0, len(c)):
            run = run + 1 if g[j, x] - c[j, x] >= 0.08 else 0
            if run == 10: best = j - 9 if best is None else min(best, j - 9); break
    return best
F = ImageFont.load_default(size=13); FB = ImageFont.load_default(size=14); props, cells = {}, []
for n, e in enumerate(EPS, 1):
    m = meta[e]; mm = np.where(ep == e)[0]; mm = mm[np.argsort(fi[mm])]; gf = min(x for x in (VB.grasp_frame(C[mm]), blocked(C[mm], G[mm]), 90) if x is not None)
    idxs = [i for i in range(0, 75, 15) if i == 0 or i < gf - 20]
    fr = frames(m, idxs); rgb = fr[0]
    for i, im in fr.items(): Image.fromarray(im).save(OUT / 'frames' / f'ep{e}_{i:04d}.jpg', quality=95)
    d = det(rgb); P = []
    for x in sorted((x for x in d if x[5] >= 0.25), key=lambda x: -x[5]):
        if all(iou(x[1:5], p['box']) < 0.6 for p in P): P.append({'box': x[1:5], 'src': 'yolo', 'cls': x[0], 'conf': x[5]})
    pr = YC.parse(instr[e]); g = diff_start(rgb, lastf(m)); tgt = None
    if g:
        c = [p for p in P if not p['cls'].startswith(('plate_', 'organizer_')) and inter(p['box'], g) / max(min(area(p['box']), area(g)), 1) > 0.5]
        if c: tgt = max(c, key=lambda p: inter(p['box'], g)); tgt['yolo_cls'] = tgt['cls']; tgt['cls'] = pr[1]; tgt['src'] = 'yolo+diff'
        else: tgt = {'box': [float(v) for v in g], 'src': 'diff', 'cls': pr[1], 'conf': None}; P.append(tgt)
    for x in sorted((x for x in d if 0.05 <= x[5] < 0.25 and not x[0].startswith(('plate_', 'organizer_'))), key=lambda x: -x[5]):
        if all(iou(x[1:5], p['box']) < 0.3 and inter(x[1:5], p['box']) / max(area(x[1:5]), 1) < 0.5 for p in P): P.append({'box': x[1:5], 'src': 'low', 'cls': None, 'hint': x[0], 'conf': x[5]})
    for b in blobs(rgb, [p['box'] for p in P]): P.append({'box': [float(v) for v in b], 'src': 'blob', 'cls': None, 'conf': None})
    for i, p in enumerate(P): p['id'] = i; p['target'] = p is tgt
    props[str(e)] = {'n': n, 'instruction': instr[e], 'target_cls': pr[1], 'dest_cls': pr[2], 'grasp_frame': gf, 'frames': sorted(fr), 'props': P}
    im = Image.fromarray(rgb); g2 = ImageDraw.Draw(im)
    for p in P:
        col = (255, 255, 255) if p['target'] else (120, 200, 255) if (p['cls'] or '').startswith(('plate_', 'organizer_')) else {'yolo': (255, 235, 0), 'low': (255, 150, 220), 'blob': (255, 60, 60), 'diff': (255, 255, 255), 'yolo+diff': (255, 255, 255)}[p['src']]
        b = p['box']; g2.rectangle(b, outline=col, width=2); lab = f"{p['id']}:{p['cls'] or '?' + (p.get('hint') or '')}" + (f" {p['conf']:.2f}" if p['conf'] else '') + (' *T*' if p['target'] else '')
        tw = g2.textlength(lab, font=F); ty = b[1] - 15 if b[1] >= 15 else b[3] + 1; g2.rectangle([b[0], ty, b[0] + tw + 4, ty + 14], fill=col); g2.text((b[0] + 2, ty), lab, font=F, fill=(0, 0, 0))
    cell = Image.new('RGB', (640, 520), (255, 255, 255)); cell.paste(im, (0, 40)); gg = ImageDraw.Draw(cell)
    gg.text((4, 2), f"[{n}] ep{e}  target={pr[1]} dest={pr[2]}  frames {sorted(fr)}" + ('' if tgt else '  TARGET NOT LOCATED'), font=FB, fill=(0, 0, 0) if tgt else (200, 0, 0)); gg.text((4, 21), instr[e][:95], font=FB, fill=(0, 0, 160)); cells.append(cell)
for k in range(0, len(cells), 4):
    cc = cells[k:k + 4]; sheet = Image.new('RGB', (1280, ((len(cc) + 1) // 2) * 520), (255, 255, 255))
    for i, c in enumerate(cc): sheet.paste(c, ((i % 2) * 640, (i // 2) * 520))
    sheet.save(OUT / f'p{k // 4 + 1:02d}.jpg', quality=82)
json.dump(props, open(OUT / 'props.json', 'w'), indent=1, ensure_ascii=False)
import collections; print('episodes', len(props), 'frames', sum(len(v['frames']) for v in props.values()), 'target located', sum(any(p['target'] for p in v['props']) for v in props.values()), collections.Counter(p['src'] for v in props.values() for p in v['props']))
