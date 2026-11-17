import json, random, sys
from pathlib import Path
from PIL import Image, ImageDraw
H = Path.home() / 'locany/label'; name = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
R = [json.loads(l) for l in open(H / name / 'labels.jsonl')]
random.seed(seed)
pick = sum((random.sample([r for r in R if r['task'] == t], 4) for t in (1, 2, 3)), [])
tiles = []
for r in pick:
    e, j = r['ep'], r['frame']
    im = Image.open(H / name / 'images' / f'ep{e:03d}' / f'{j:04d}.jpg').convert('RGB'); d = ImageDraw.Draw(im)
    if r['task'] < 3:
        if r.get('target_box'): d.rectangle(r['target_box'], outline=(255, 0, 0), width=3)
        d.text((5, 5), f"ep{e} f{j}: {r['target']}", fill=(255, 255, 0))
    else:
        for b in r['boxes']:
            d.rectangle(b[1:], outline=(255, 0, 0), width=2); d.text((b[1] + 2, b[2] + 2), b[0], fill=(255, 255, 0))
        d.text((5, 5), f'ep{e} f{j}', fill=(255, 255, 0))
    tiles.append(im.resize((400, 300)))
M = Image.new('RGB', (1600, 900))
for i, t in enumerate(tiles): M.paste(t, ((i % 4) * 400, (i // 4) * 300))
M.save(H / f'{name}_check.jpg', quality=88)
