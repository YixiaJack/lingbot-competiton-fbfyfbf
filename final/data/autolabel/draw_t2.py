"""Draw YOLO v1 frame-0 detections on task-2 episodes where the target-class box is on the right object (diffgt)."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
S = Path(sys.argv[1]); IMG = Path(sys.argv[2]); OUT = Path(sys.argv[3]); RUN = sys.argv[4] if len(sys.argv) > 4 else 'v1'; TH = 0.25
OUT.mkdir(parents=True, exist_ok=True)
D = json.load(open(S / f'dets_{RUN}.json')); rows = {r['ep']: r for r in json.load(open(S / f'dets_{RUN}_rows.json'))}
G = {int(k): v for k, v in json.load(open(S / 'diffgt.json')).items()}
idx = {r['ep']: r for r in json.load(open(S / 'index.json'))}
F = ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc', 15); FB = ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc', 17)
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)
ctr = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2); inside = lambda p, b: b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]
match = lambda b, g: iou(b, g) >= 0.3 or (inside(ctr(b), g) and inside(ctr(g), b))
def good(g):
    s = g.get('start_box')
    return s and (s[2] - s[0]) * (s[3] - s[1]) < 20000 and not (g.get('end_box') and iou(s, g['end_box']) > 0)
def label(d, xy, text, color):
    x, y = xy; w = d.textlength(text, font=F)
    d.rectangle([x, y - 18, x + w + 6, y], fill=color); d.text((x + 3, y - 18), text, font=F, fill=(0, 0, 0))
ok, thumbs = [], []
for e in sorted(rows):
    r = rows[e]
    if r['t'] != 2 or e not in G or not good(G[e]): continue
    dets = [x for x in D['first/%04d' % e] if x[5] >= TH]
    tb = [x for x in dets if x[0] == r['tgt']]
    if not tb or not match(max(tb, key=lambda x: x[5])[1:5], G[e]['start_box']): continue
    t = max(tb, key=lambda x: x[5]); db = [x for x in dets if x[0] == r['dest']]
    dest_ok = bool(db) and 'end_box' in G[e] and inside(ctr(G[e]['end_box']), max(db, key=lambda x: x[5])[1:5])
    im = Image.open(IMG / f'ep{e:03d}/0000.jpg').convert('RGB'); W, H = im.size
    canvas = Image.new('RGB', (W, H + 64), (255, 255, 255)); canvas.paste(im, (0, 64)); d = ImageDraw.Draw(canvas)
    reserve = 'reserve' in idx[e]['id']
    d.text((6, 4), f"ep{e}  {idx[e]['id']}{'（reserve，没训过）' if reserve else ''}  目的地{'对' if dest_ok else '错/没框到'}", font=FB, fill=(0, 0, 0))
    d.text((6, 28), idx[e]['instr'][:88], font=F, fill=(60, 60, 60))
    off = lambda b: [b[0], b[1] + 64, b[2], b[3] + 64]
    for x in dets:   # other boxes first, thin
        if x is t or (db and x is max(db, key=lambda y: y[5])): continue
        d.rectangle(off(x[1:5]), outline=(120, 180, 255), width=1); label(d, (x[1], x[2] + 64), f'{x[0]} {x[5]:.2f}', (170, 210, 255))
    d.rectangle(off(G[e]['start_box']), outline=(0, 200, 0), width=1)
    if db:
        b = max(db, key=lambda y: y[5]); d.rectangle(off(b[1:5]), outline=(255, 200, 0), width=3); label(d, (b[1], b[2] + 64), f'目的地 {b[0]} {b[5]:.2f}', (255, 220, 80))
    d.rectangle(off(t[1:5]), outline=(255, 0, 0), width=3); label(d, (t[1], t[2] + 64), f'目标 {t[0]} {t[5]:.2f}', (255, 120, 120))
    name = f"{len(ok) + 1:02d}_ep{e}_{r['tgt']}{'_reserve' if reserve else ''}.jpg"
    canvas.save(OUT / name, quality=92); ok.append((e, r['tgt'], dest_ok, reserve)); thumbs.append(canvas.resize((320, 272)))
cols = 6; rws = (len(thumbs) + cols - 1) // cols
M = Image.new('RGB', (cols * 320, rws * 272), (255, 255, 255))
for i, th in enumerate(thumbs): M.paste(th, ((i % cols) * 320, (i // cols) * 272))
M.save(OUT / '00_总览.jpg', quality=85)
from collections import Counter
c = Counter(x[1] for x in ok)
(OUT / '说明.txt').write_text(
    f"YOLO {RUN}（公开 yolo11s 重训）在任务 2 第 0 帧上的成功样本：目标类别分数最高的框（阈值 {TH}）落在真目标上。\n"
    f"共 {len(ok)} 条，其中 reserve（没参与训练）{sum(x[3] for x in ok)} 条；这些里目的地也对的 {sum(x[2] for x in ok)} 条。\n"
    f"按目标物：{dict(c.most_common())}\n\n"
    "图例：红框 = 目标类别的最高分框；黄框 = 目的地类别（收纳盒某层）的最高分框；绿色细框 = 首末帧差分得到的目标真实位置；"
    "浅蓝细框 = 其他检测（≥0.25）。框上数字是置信度。\n"
    "注意：大多数是训练过的条（训练帧里有这一条的第 0 帧附近），所以偏乐观；reserve 的才是没见过的。\n")
print(len(ok), sum(x[3] for x in ok), sum(x[2] for x in ok), dict(c))
