"""v13 labels for today's live raw frames (yolo13/live_prop/props.json + yolo13/fix_live.json): yolo proposals keep their
class unless fixed, low/blob proposals only when named, 'drop' removes. Held out: 175700 (pomegranate next to red
delicious) and the pear+mango scene 172549/173022. Writes yolo13/ds/images|labels/train (l13_*), holdout list, sheet."""
import json, collections
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
H = Path('yolo13'); P = json.load(open(H / 'live_prop/props.json')); FX = json.load(open(H / 'fix_live.json'))
names = json.load(open('weights/v12.names.json'))['names']; nid = {n: i for i, n in enumerate(names)}
HOLD = {'20261115_175700', '20261115_172549', '20261115_173022'}
O = H / 'ds'; (O / 'images/train').mkdir(parents=True, exist_ok=True); (O / 'labels/train').mkdir(parents=True, exist_ok=True)
F = ImageFont.load_default(size=13); final = {}; cells = []; st = collections.Counter()
for b, v in sorted(P.items()):
    key = b[:15]; fx = FX.get(key, {}); labels = []
    for p in v['props']:
        if p['id'] in fx.get('drop', []): continue
        c = fx.get('set', {}).get(str(p['id']))
        if c is None:
            if p['src'] in ('low', 'blob'): continue
            c = p['cls']
        assert c in nid, (b, c); labels.append((c, p['box']))
    final[b] = labels; im = Image.open(H / f'live/{b}_nocue_raw.jpg').convert('RGB'); W, Hh = im.size
    txt = ''.join(f"{nid[c]} {(x[0] + x[2]) / 2 / W:.6f} {(x[1] + x[3]) / 2 / Hh:.6f} {(x[2] - x[0]) / W:.6f} {(x[3] - x[1]) / Hh:.6f}\n" for c, x in labels)
    if key in HOLD: st['holdout'] += 1
    else:
        im.save(O / 'images/train' / f'l13_{b}.jpg', quality=95); (O / 'labels/train' / f'l13_{b}.txt').write_text(txt); st['train'] += 1
    g = ImageDraw.Draw(im)
    for c, x in labels:
        col = (120, 200, 255) if c.startswith(('plate_', 'organizer_')) else (255, 235, 0)
        g.rectangle(x, outline=col, width=2); tw = g.textlength(c, font=F); ty = x[1] - 15 if x[1] >= 15 else x[3] + 1
        g.rectangle([x[0], ty, x[0] + tw + 4, ty + 14], fill=col); g.text((x[0] + 2, ty), c, font=F, fill=(0, 0, 0))
    cell = Image.new('RGB', (640, 500), 'white'); cell.paste(im, (0, 20)); ImageDraw.Draw(cell).text((4, 3), f"{key[9:]} {'HOLDOUT ' if key in HOLD else ''}{v['task'][:70]}", font=F, fill=(200, 0, 0) if key in HOLD else (0, 0, 160)); cells.append(cell)
for k in range(0, len(cells), 6):
    cc = cells[k:k + 6]; sh = Image.new('RGB', (1920, ((len(cc) + 2) // 3) * 500), 'white')
    for i, c in enumerate(cc): sh.paste(c, ((i % 3) * 640, (i // 3) * 500))
    sh.save(H / f'final_live_{k // 6 + 1}.jpg', quality=85)
json.dump({'holdout': sorted(HOLD), 'labels': final}, open(O / 'live_labels.json', 'w'), indent=1); print(dict(st))
