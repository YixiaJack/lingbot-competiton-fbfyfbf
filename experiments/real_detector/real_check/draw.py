import sys, json
from PIL import Image, ImageDraw
D = json.load(open(sys.argv[1])); rows = {r['ep']: r for r in json.load(open(sys.argv[1].replace('.json', '_rows.json')))}
idx = {r['ep']: r for r in json.load(open('index.json'))}
eps = [int(e) for e in sys.argv[3].split(',')]; th = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
W, H = 640, 480; g = Image.new('RGB', (W * 2, (H + 30) * ((len(eps) + 1) // 2)), 'white')
for i, ep in enumerate(eps):
    im = Image.open(f'first/{ep:04d}.png').convert('RGB'); dr = ImageDraw.Draw(im)
    r = rows[ep]
    for c, x0, y0, x1, y1, s in D[f'first/{ep:04d}']:
        if s < th: continue
        col = 'red' if c == r['tgt'] else 'yellow' if c == r['dest'] else 'white'
        dr.rectangle([x0, y0, x1, y1], outline=col, width=3); dr.text((x0 + 3, y0 + 2), f'{c} {s:.2f}', fill=col)
    x, y = (i % 2) * W, (i // 2) * (H + 30); g.paste(im, (x, y + 30))
    ImageDraw.Draw(g).text((x + 5, y + 8), f"ep{ep}: {idx[ep]['instr'][:90]}", fill='black')
g.save(sys.argv[2], quality=85)
