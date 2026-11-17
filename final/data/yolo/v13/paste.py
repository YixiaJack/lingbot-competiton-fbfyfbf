"""v13 copy-paste images: verified crops (train split only) of pear / red_apple / red_delicious / pomegranate / tomato /
persimmon, cut out with GrabCut, pasted onto verified v13 train backgrounds right next to a look-alike (pear next to
potato / lemon / mango / starfruit / papaya; the red fruits next to each other, sometimes touching). Brightness matched to
the table at the landing spot, scaled with depth, never over another labelled box (touching allowed up to 8% overlap
with the anchor). Labels = background labels + pasted boxes. Writes yolo13/ds/images|labels/train (p13_*) + sample sheet."""
import json, random
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
H = Path.home() / 'lingbot_v7work/yolo13'; O = H / 'ds'
names = json.load(open(Path.home() / 'lingbot_v7work/weights/v12.names.json'))['names']; nid = {n: i for i, n in enumerate(names)}
own = json.load(open(O / 'final_labels.json')); live = json.load(open(O / 'live_labels.json'))
FOCUS = {'pear', 'red_apple', 'red_delicious', 'pomegranate', 'tomato', 'persimmon'}
NEAR = {'pear': {'potato', 'lemon', 'mango', 'starfruit', 'papaya', 'banana', 'pineapple'},
        'red_apple': {'red_delicious', 'pomegranate', 'tomato', 'bell_pepper', 'wax_apple', 'persimmon'},
        'red_delicious': {'red_apple', 'pomegranate', 'tomato', 'wax_apple', 'mangosteen', 'onion'},
        'pomegranate': {'red_delicious', 'red_apple', 'tomato', 'persimmon', 'peach'},
        'tomato': {'red_apple', 'persimmon', 'orange', 'pumpkin', 'pomegranate', 'red_delicious'},
        'persimmon': {'tomato', 'orange', 'pumpkin', 'tangerine', 'pomegranate'}}
# backgrounds (train split): image path + labels
bgs = []
for e, labs in own['labels'].items():
    if int(e) in set(own['holdout']): continue
    p = O / 'images/train' / f'n13_ep{e}_0000.jpg'
    if p.exists(): bgs.append((p, labs))
for b, labs in live['labels'].items():
    if b[:15] in set(live['holdout']): continue
    bgs.append((O / 'images/train' / f'l13_{b}.jpg', labs))
# crops
def grab(img, box):
    x0, y0, x1, y1 = [int(round(v)) for v in box]; m = 6
    X0, Y0, X1, Y1 = max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, img.shape[1]), min(y1 + m, img.shape[0])
    sub = img[Y0:Y1, X0:X1].copy(); mask = np.zeros(sub.shape[:2], np.uint8)
    rect = (x0 - X0, y0 - Y0, max(x1 - x0, 2), max(y1 - y0, 2)); bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(sub, mask, rect, bg, fg, 5, cv2.GC_INIT_WITH_RECT)
    a = np.where((mask == 1) | (mask == 3), 1.0, 0.0).astype(np.float32)
    a = cv2.GaussianBlur(cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), (0, 0), 1.2)
    ys, xs = np.where(a > 0.5)
    if len(ys) < 50: return None
    t, b_, l, r = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return {'rgb': sub[t:b_, l:r], 'a': a[t:b_, l:r], 'ybot': Y0 + b_, 'tab': table_v(img, (X0 + l, Y0 + b_))}
def table_v(img, pt):
    x, y = int(pt[0]), int(min(pt[1] + 4, img.shape[0] - 1)); patch = img[max(y - 6, 0):y + 6, max(x - 40, 0):x + 40]
    return float(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[..., 2].mean()) if patch.size else 128.0
crops = {c: [] for c in FOCUS}
for p, labs in bgs:
    img = cv2.imread(str(p))
    for c, box in labs:
        if c in FOCUS and not (c == 'pear' and 'ep22_' in p.name):   # the lying pear of ep22 looks like a round orange: real frame kept, not pasted
            g = grab(img, box)
            if g: g['src'] = p.name; crops[c].append(g)
print({c: len(v) for c, v in crops.items()})
rng = random.Random(13)
area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1]); inter = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
def table_top(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); col = np.median(hsv[int(img.shape[0] * .7):, :, 0]); rows = (np.abs(hsv[..., 0].astype(int) - col) < 8).mean(1)
    return next((y for y in range(img.shape[0] // 5, img.shape[0] - 20) if rows[y:y + 10].mean() > 0.6), int(img.shape[0] * .4))
def paste(img, labs, cls):
    anchors = [(c, b) for c, b in labs if c in NEAR[cls]]
    if not anchors or not crops[cls]: return None
    ac, ab = rng.choice(anchors); cr = rng.choice(crops[cls]); H0, W0 = img.shape[:2]; top = table_top(img)
    for _ in range(40):
        side = rng.choice(['l', 'r', 'b', 'bl', 'br']); gap = rng.choice([-6, -3, 0, 2, 5, 10])
        s = np.clip(((ab[3] + 30) / max(cr['ybot'], 1)) ** 0.8, 0.8, 1.25) * rng.uniform(0.92, 1.08)
        h, w = int(cr['rgb'].shape[0] * s), int(cr['rgb'].shape[1] * s)
        if h < 12 or w < 12: continue
        if side == 'l': x0, yb = ab[0] - w - gap, ab[3] + rng.randint(-8, 8)
        elif side == 'r': x0, yb = ab[2] + gap, ab[3] + rng.randint(-8, 8)
        elif side == 'b': x0, yb = (ab[0] + ab[2]) / 2 - w / 2 + rng.randint(-15, 15), ab[3] + h + gap
        elif side == 'bl': x0, yb = ab[0] - w * 0.6, ab[3] + h * 0.7 + gap
        else: x0, yb = ab[2] - w * 0.4, ab[3] + h * 0.7 + gap
        box = [int(x0), int(yb - h), int(x0) + w, int(yb)]
        if box[0] < 2 or box[2] > W0 - 2 or box[1] < top + 4 or box[3] > H0 - 2: continue
        if any(inter(box, b) > (0.08 * min(area(box), area(b)) if (c, b) == (ac, ab) else 0) for c, b in labs): continue
        rgb = cv2.resize(cr['rgb'], (w, h), interpolation=cv2.INTER_AREA).astype(np.float32); a = cv2.resize(cr['a'], (w, h))[..., None]
        if rng.random() < 0.5: rgb, a = np.ascontiguousarray(rgb[:, ::-1]), np.ascontiguousarray(a[:, ::-1])
        k = np.clip(table_v(img, (box[0] + w / 2, box[3])) / max(cr['tab'], 1), 0.8, 1.2); rgb = np.clip(rgb * k, 0, 255)
        roi = img[box[1]:box[3], box[0]:box[2]].astype(np.float32)
        sh = np.zeros((h, w), np.float32); sh[int(h * 0.75):, :] = 0.18; sh = cv2.GaussianBlur(sh, (0, 0), 3)[..., None] * (1 - a)   # soft contact shadow
        roi = roi * (1 - sh) ; img[box[1]:box[3], box[0]:box[2]] = (a * rgb + (1 - a) * roi).astype(np.uint8)
        ys, xs = np.where(a[..., 0] > 0.5); tb = [box[0] + int(xs.min()), box[1] + int(ys.min()), box[0] + int(xs.max()) + 1, box[1] + int(ys.max()) + 1]
        return (cls, [float(v) for v in tb]), f'{cls}@{ac}:{side}'
    return None
N = 240; made = 0; plan = ['pear'] * 3 + ['red_apple', 'red_delicious', 'pomegranate', 'tomato', 'persimmon']
samples = []; F = ImageFont.load_default(size=13)
while made < N:
    p, labs = rng.choice(bgs); img = cv2.imread(str(p)); labs = [(c, list(b)) for c, b in labs]; notes = []
    for _ in range(rng.choice([1, 2, 2, 3])):
        r = paste(img, labs, rng.choice(plan))
        if r: labs.append(r[0]); notes.append(r[1])
    if not notes: continue
    name = f'p13_{made:04d}'; cv2.imwrite(str(O / 'images/train' / f'{name}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    Hh, Ww = img.shape[:2]
    (O / 'labels/train' / f'{name}.txt').write_text(''.join(f"{nid[c]} {(b[0] + b[2]) / 2 / Ww:.6f} {(b[1] + b[3]) / 2 / Hh:.6f} {(b[2] - b[0]) / Ww:.6f} {(b[3] - b[1]) / Hh:.6f}\n" for c, b in labs))
    if len(samples) < 24:
        im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); g = ImageDraw.Draw(im)
        for c, b in labs[-len(notes):]: g.rectangle(b, outline=(255, 0, 255), width=2); g.text((b[0] + 2, b[1] - 14 if b[1] > 14 else b[3] + 1), c, font=F, fill=(255, 0, 255))
        g.text((4, 4), f'{name} {p.name[:24]} | ' + ', '.join(notes), font=F, fill=(255, 255, 255)); samples.append(im)
    made += 1
for k in range(0, len(samples), 6):
    cc = samples[k:k + 6]; sh = Image.new('RGB', (1920, 960), 'white')
    for i, c in enumerate(cc): sh.paste(c, ((i % 3) * 640, (i // 3) * 480))
    sh.save(H / f'paste_sample_{k // 6 + 1}.jpg', quality=85)
import collections
print('made', made, collections.Counter(l.split()[0] for f in (O / 'labels/train').glob('p13_*.txt') for l in open(f) if names[int(l.split()[0])] in FOCUS and True))
