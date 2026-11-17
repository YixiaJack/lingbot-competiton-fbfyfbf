import json, sys
from pathlib import Path
import av, cv2, numpy as np, pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont, ImageFilter
sys.path.insert(0, '.')
import yolocue_bridge as YC, v7_boxes as VB
V1 = Path.home() / 'Downloads/b60f1e2e_0d3bf9f6/v1'
meta = {m['episode_index']: m for m in pq.read_table(sorted((V1 / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()}
Cd = {int(k): v for k, v in json.load(open('out/extra_cands.json')).items()}
det = YC.Detector(YC.ROOT / 'weights/v7a.torchscript', json.loads((YC.ROOT / 'weights/v7a.names.json').read_text())['names'], conf=0.05)
area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1]); inter = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
CAM = VB.CAM
def frames(m, idx):
    p = V1 / 'videos' / CAM / f"chunk-{m[f'videos/{CAM}/chunk_index']:03d}" / f"file-{m[f'videos/{CAM}/file_index']:03d}.mp4"; t0 = m[f'videos/{CAM}/from_timestamp']
    want = sorted(idx); got = {}; ts = [t0 + i / 30 for i in want]
    with av.open(str(p)) as ic:
        s = ic.streams.video[0]; ic.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s, backward=True); k = 0
        for fr in ic.decode(s):
            while k < len(want) and float(fr.pts * s.time_base) + 1e-3 >= ts[k]: got[want[k]] = fr.to_ndarray(format='rgb24'); k += 1
            if k == len(want): break
    return got
def blobs(a, b):
    A = np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(2)), np.float32); B = np.asarray(Image.fromarray(b).filter(ImageFilter.GaussianBlur(2)), np.float32)
    H = A.shape[0]; m = (np.abs(A - B).max(2) > 45).astype(np.uint8); m[:int(H * .25)] = 0
    k = np.ones((3, 3), np.uint8); m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2), cv2.MORPH_CLOSE, k, iterations=4)
    low = np.zeros_like(m, bool); low[H // 2:] = True; tab = np.median(A[low & (m == 0)], 0); n, lab, st, _ = cv2.connectedComponentsWithStats(m); out = []
    for i in range(1, n):
        x, y, w, h, ar = [int(v) for v in st[i]]
        if ar < 150: continue
        sl = (slice(y, y + h), slice(x, x + w)); comp = lab[sl] == i
        g = float((np.linalg.norm(B[sl][comp] - tab, axis=1) < 35).mean()) - float((np.linalg.norm(A[sl][comp] - tab, axis=1) < 35).mean())
        out.append({'box': [x, y, x + w, y + h], 'area': ar, 'gain': g})
    return out
F = ImageFont.load_default(size=13); cells = []; props = {}
use = [e for e, v in sorted(Cd.items()) if v['grasp_frame'] is not None and len(v['arms']) == 1]
for n, e in enumerate(use, 1):
    v = Cd[e]; m = meta[e]; gf = v['grasp_frame']; task, tcls, dcls = v['parsed']
    fr = frames(m, [0, gf, v['T'] - 3]); a, gimg, b = fr[0], fr[gf], fr[v['T'] - 3]; d = det(a)
    tb, db, reason = YC.select(d, task, tcls, dcls, rgb=a); how = 'yolo'
    if reason != 'ok':
        B = blobs(a, b); st = max(B, key=lambda c: c['gain'] * np.sqrt(c['area'])) if B else None
        if tb is None and st and st['gain'] > 0.2:
            g = st['box']; c = [x for x in d if not x[0].startswith(('plate_', 'organizer_')) and x[5] >= 0.05 and inter(x[1:5], g) / max(min(area(x[1:5]), area(g)), 1) > 0.5]
            if c: best = max(c, key=lambda x: inter(x[1:5], g) / max(area(x[1:5]), area(g))); tb = [round(x, 1) for x in best[1:5]]; how = f'diff+det {best[0]} {best[5]:.2f}'
            else: tb = [float(x) for x in g]; how = 'diff blob'
        if db is None:
            cand = [x for x in d if x[0] == dcls and x[5] >= 0.1]
            if cand: x = max(cand, key=lambda x: x[5]); db = [round(y, 1) for y in x[1:5]]; how += f' | dest {x[0]} {x[5]:.2f}'
    props[e] = {'id': f'ep{e:04d}', 'instruction': v['instruction'], 'target': tcls, 'dest': dcls, 'target_box': tb, 'dest_box': db, 'grasp_frame': gf, 'T': v['T'], 'how': how, 'rule': v['rule']}
    p0 = Image.fromarray(YC.draw(a, tb, db, False)) if tb and db else Image.fromarray(a)
    pg = Image.fromarray(gimg); g2 = ImageDraw.Draw(pg)
    if tb: g2.rectangle(tb, outline=(255, 255, 255), width=2)
    cell = Image.new('RGB', (1440, 380), (255, 255, 255))
    for k, im in enumerate((p0, pg, Image.fromarray(b))): cell.paste(im.resize((480, 360)), (k * 480, 20))
    ImageDraw.Draw(cell).text((4, 3), f"#{n} ep{e} {tcls}->{dcls} | {how} | rule {v['rule'] or 'ok'} | {v['instruction'][:70]} | mid: grasp frame {gf}, right: last", font=F, fill=(0, 0, 0) if tb and db else (200, 0, 0))
    cells.append(cell.resize((1200, 317)))
for k in range(0, len(cells), 5):
    cc = cells[k:k + 5]; sheet = Image.new('RGB', (1200, len(cc) * 317), (255, 255, 255))
    for i, c in enumerate(cc): sheet.paste(c, (0, i * 317))
    sheet.save(f'out/extra_{k // 5 + 1}.jpg', quality=85)
json.dump({str(k): v for k, v in props.items()}, open('out/extra_props.json', 'w'), indent=1, ensure_ascii=False)
print(len(use), 'rendered;', sum(1 for p in props.values() if p['target_box'] and p['dest_box']), 'with a full cue')
