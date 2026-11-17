import json, sys
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '.')
import yolocue_bridge as YC, v7_boxes as VB
V1 = Path.home() / 'Downloads/b60f1e2e_cbd7865e/v1'
meta = {m['episode_index']: m for m in pq.read_table(sorted((V1 / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()}
new = set(int(k) for k in json.load(open('new_t12.json')))
F = ImageFont.load_default(size=13); cells = []; index = []
for t in (1, 2):
    for k, r in sorted(json.load(open(f'out/own_v7a_t{t}.json'))['episodes'].items(), key=lambda kv: int(kv[0])):
        e = int(k); im = Image.fromarray(YC.draw(VB.frame0(V1, meta[e]), r['target_box'], r['dest_box'], False)).resize((400, 300), Image.LANCZOS)
        c = Image.new('RGB', (400, 334), (255, 255, 255)); c.paste(im, (0, 34)); g = ImageDraw.Draw(c)
        n = len(cells) + 1; index.append((n, e, r['target'], r['dest']))
        g.text((3, 1), f"{n}. ep{e}{' NEW' if e in new else ''}  {r['target']} -> {r['dest']}", font=F, fill=(200, 0, 0) if e in new else (0, 0, 0))
        g.text((3, 17), r['instruction'][:62], font=F, fill=(0, 0, 160)); cells.append(c)
for s in range(0, len(cells), 20):
    cc = cells[s:s + 20]; sheet = Image.new('RGB', (2000, ((len(cc) + 4) // 5) * 334), (255, 255, 255))
    for i, c in enumerate(cc): sheet.paste(c, ((i % 5) * 400, (i // 5) * 334))
    sheet.save(f'out/ownchk_{s // 20 + 1}.jpg', quality=85)
json.dump(index, open('out/ownchk_index.json', 'w'))
print(len(cells))
