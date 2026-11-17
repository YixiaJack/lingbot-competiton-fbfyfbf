"""Sheets of the suspect official images: labels in yellow (plates/organizer blue), suspects in red with the reason; numbered."""
import json, collections
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
R = Path('mine'); D = json.load(open(R / 'home/guest/yolocue/test_out/mine_v6.json'))
F = ImageFont.load_default(size=12); FB = ImageFont.load_default(size=14); cells = []; index = []
for n, (p, v) in enumerate(sorted(D.items(), key=lambda kv: (kv[1]['bad'][0]['t'], kv[0])), 1):
    im = Image.open(R / p.lstrip('/')).convert('RGB'); g = ImageDraw.Draw(im)
    for c, b in v['labels']:
        col = (120, 200, 255) if c.startswith(('plate_', 'organizer_')) else (255, 235, 0); g.rectangle(b, outline=col, width=2); g.text((b[0] + 2, b[1] + 1), c, font=F, fill=col)
    tags = []
    for i, x in enumerate(v['bad']):
        b = x['box']; g.rectangle(b, outline=(255, 40, 40), width=3)
        t = f"{chr(97 + i)}:{x['t']} " + (f"{x['label']}->{x['v9']} {x['conf']}" if x['t'] == 'cls' else f"{x['v9']} {x['conf']}" if x['t'] == 'miss' else x['label'])
        g.rectangle([b[0], b[3], b[0] + g.textlength(t, font=F) + 4, b[3] + 13], fill=(255, 40, 40)); g.text((b[0] + 2, b[3]), t, font=F, fill='white'); tags.append(t)
    cell = Image.new('RGB', (480, 380), 'white'); cell.paste(im.resize((480, 360)), (0, 20))
    ImageDraw.Draw(cell).text((3, 3), f'[{n}] {Path(p).stem}', font=FB, fill='black'); cells.append(cell); index.append({'n': n, 'img': p, 'bad': v['bad']})
for k in range(0, len(cells), 16):
    cc = cells[k:k + 16]; sh = Image.new('RGB', (1920, ((len(cc) + 3) // 4) * 380), 'white')
    for i, c in enumerate(cc): sh.paste(c, ((i % 4) * 480, (i // 4) * 380))
    sh.save(f'mine_{k // 16 + 1:02d}.jpg', quality=82)
json.dump(index, open('mine_index.json', 'w'), indent=1); print(len(cells), 'cells,', (len(cells) + 15) // 16, 'sheets')
