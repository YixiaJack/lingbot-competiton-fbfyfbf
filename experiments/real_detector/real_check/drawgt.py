import json, random, sys
from PIL import Image, ImageDraw
G = {int(k): v for k, v in json.load(open('diffgt.json')).items()}; idx = {r['ep']: r for r in json.load(open('index.json'))}
eps = [int(e) for e in sys.argv[2].split(',')] if len(sys.argv) > 2 else random.Random(0).sample(sorted(G), 12)
W, H = 400, 300; g = Image.new('RGB', (W * 4, (H + 22) * ((len(eps) + 1) // 2)), 'white')
for i, ep in enumerate(eps):
    for j, which in enumerate(('first', 'last')):
        im = Image.open(f'{which}/{ep:04d}.png').convert('RGB'); dr = ImageDraw.Draw(im)
        if 'start_box' in G[ep]: dr.rectangle(G[ep]['start_box'], outline='lime', width=3)
        if 'end_box' in G[ep]: dr.rectangle(G[ep]['end_box'], outline='blue', width=3)
        x, y = (i % 2) * 2 * W + j * W, (i // 2) * (H + 22); g.paste(im.resize((W, H)), (x, y + 22))
    ImageDraw.Draw(g).text(((i % 2) * 2 * W + 4, (i // 2) * (H + 22) + 5), f"ep{ep} {idx[ep]['instr'][:85]}", fill='black')
g.save(sys.argv[1], quality=82); print(eps)
