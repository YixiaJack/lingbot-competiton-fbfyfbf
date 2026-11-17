"""Final YOLO v9 labels for the 38 newest own episodes (264-301, task 1 + 2): proposals + hand fixes (yolo9/fix.json; "drop" = not a label, pixels kept). Same label on every kept frame
of an episode (pre-grasp, scene static); a frame is dropped if any labelled box changed vs frame 0 (arm occlusion).
Low boxes / blobs become labels only when fix.json names them; 'unknown' ones are painted with the table colour.
Every third episode (sorted) is held out. Writes yolo9/ds/{images,labels}/train + holdout list + review sheets."""
import json, collections
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
H8 = Path('yolo9'); P = json.load(open(H8 / 'prop/props.json')); FX = json.load(open(H8 / 'fix.json'))
names = json.load(open('weights/v8.names.json'))['names']; nid = {n: i for i, n in enumerate(names)}
eps = sorted(int(e) for e in P); hold = set(eps[2::3])
O = H8 / 'ds'; (O / 'images/train').mkdir(parents=True, exist_ok=True); (O / 'labels/train').mkdir(parents=True, exist_ok=True); (O / 'holdout').mkdir(exist_ok=True)
F = ImageFont.load_default(size=13); cells = []; st = collections.Counter(); final = {}
for e in eps:
    v = P[str(e)]; fx = FX.get(str(e), {}); labels = []; erase = []
    for p in v['props']:
        c = fx.get('set', {}).get(str(p['id']))
        if p['id'] in fx.get('unknown', []): erase.append(p['box']); continue
        if p['id'] in fx.get('drop', []): st['dropped'] += 1; continue
        if c is None:
            if p['src'] in ('low', 'blob'): continue
            c = p['cls']
        else: st['fixed'] += 1
        assert c in nid, (e, c); labels.append((c, [float(x) for x in p['box']]))
    final[e] = labels
    f0 = cv2.imread(str(H8 / f'prop/frames/ep{e}_0000.jpg')); Hh, Ww = f0.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(f0, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32); tabpix = f0[int(Hh * .6):].reshape(-1, 3); col = np.median(tabpix, 0)
    txt = ''.join(f"{nid[c]} {(b[0] + b[2]) / 2 / Ww:.6f} {(b[1] + b[3]) / 2 / Hh:.6f} {(b[2] - b[0]) / Ww:.6f} {(b[3] - b[1]) / Hh:.6f}\n" for c, b in labels)
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY).astype(np.float32); kept = []
    for fr in v['frames']:
        im = cv2.imread(str(H8 / f'prop/frames/ep{e}_{fr:04d}.jpg'))
        if fr:
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
            worst = max(float(np.abs(g[int(b[1]):int(b[3]), int(b[0]):int(b[2])] - g0[int(b[1]):int(b[3]), int(b[0]):int(b[2])]).mean()) for c, b in labels)
            if worst > 12: st['frames dropped (box changed)'] += 1; continue
        if erase:
            m = np.zeros((Hh, Ww), np.float32)
            for b in erase: m[max(int(b[1]) - 5, 0):int(b[3]) + 5, max(int(b[0]) - 5, 0):int(b[2]) + 5] = 1
            a = cv2.GaussianBlur(m, (0, 0), 2.0)[..., None]; im = np.clip(a * col + (1 - a) * im, 0, 255).astype(np.uint8)
        split = 'holdout' if e in hold else 'train'
        if split == 'train':
            cv2.imwrite(str(O / 'images/train' / f'n9_ep{e}_{fr:04d}.jpg'), im, [cv2.IMWRITE_JPEG_QUALITY, 95]); (O / 'labels/train' / f'n9_ep{e}_{fr:04d}.txt').write_text(txt)
        kept.append(fr); st[f'frames {split}'] += 1
    im = Image.fromarray(cv2.cvtColor(cv2.imread(str(H8 / f'prop/frames/ep{e}_0000.jpg')), cv2.COLOR_BGR2RGB)); g = ImageDraw.Draw(im)
    for c, b in labels:
        colr = (120, 200, 255) if c.startswith(('plate_', 'organizer_')) else (255, 235, 0)
        g.rectangle(b, outline=colr, width=2); tw = g.textlength(c, font=F); ty = b[1] - 15 if b[1] >= 15 else b[3] + 1
        g.rectangle([b[0], ty, b[0] + tw + 4, ty + 14], fill=colr); g.text((b[0] + 2, ty), c, font=F, fill=(0, 0, 0))
    for b in erase: g.rectangle(b, outline=(255, 0, 0), width=2)
    cell = Image.new('RGB', (640, 500), (255, 255, 255)); cell.paste(im, (0, 20))
    ImageDraw.Draw(cell).text((4, 3), f"ep{e} {'HOLDOUT ' if e in hold else ''}frames {kept} | {v['instruction'][:70]}", font=F, fill=(0, 0, 160) if e not in hold else (200, 0, 0)); cells.append(cell)
for k in range(0, len(cells), 6):
    cc = cells[k:k + 6]; sheet = Image.new('RGB', (1920, ((len(cc) + 2) // 3) * 500), (255, 255, 255))
    for i, c in enumerate(cc): sheet.paste(c, ((i % 3) * 640, (i // 3) * 500))
    sheet.save(H8 / f'final_{k // 6 + 1}.jpg', quality=85)
json.dump({'holdout': sorted(hold), 'labels': {str(k): v for k, v in final.items()}}, open(O / 'final_labels.json', 'w'), indent=1)
print(dict(st), 'holdout eps', sorted(hold))
print('object labels by class:', collections.Counter(c for v in final.values() for c, b in v if not c.startswith(('plate_', 'organizer_'))).most_common())
