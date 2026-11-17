"""Ground truth from first/last frames (arms are home at the end, only the target moved):
the changed blob whose last-frame pixels look like bare table = where the target started (box in the first frame);
the other big blob = where it ended (inside the destination)."""
import json, sys
import numpy as np
from PIL import Image, ImageFilter
import cv2
idx = json.load(open('index.json'))
def load(p): return np.asarray(Image.open(p).convert('RGB').filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
out = {}
for r in idx:
    if not r['id'].startswith(('T1', 'T2')): continue
    ep = r['ep']; a = load(f'first/{ep:04d}.png'); b = load(f'last/{ep:04d}.png'); H, W = a.shape[:2]
    d = np.abs(a - b).max(2); m = d > 45; m[:int(H * 0.25)] = False          # top quarter is the backdrop
    k = np.ones((3, 3), np.uint8); m8 = m.astype(np.uint8)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k, iterations=2); m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k, iterations=4)
    cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); m8 = np.zeros_like(m8); cv2.drawContours(m8, cs, -1, 1, -1)
    m = m8.astype(bool); n, lab, stats, _ = cv2.connectedComponentsWithStats(m8)
    # table colour: median of the lower-half pixels that did not change
    low = np.zeros_like(m); low[H // 2:] = True
    tab = np.median(a[low & ~m], axis=0)
    def teal(img, sl, comp): return float((np.linalg.norm(img[sl][comp] - tab, axis=1) < 35).mean())
    comps = []
    for i in range(1, n):
        x0, y0, w, h, area = [int(v) for v in stats[i]]
        if area < 250: continue
        x1, y1 = x0 + w, y0 + h; sl = (slice(y0, y1), slice(x0, x1)); comp = lab[sl] == i
        comps.append(dict(box=[x0, y0, x1, y1], area=area, teal_first=teal(a, sl, comp), teal_last=teal(b, sl, comp)))
    rec = {'n_blobs': len(comps)}
    if comps:
        orig = max(comps, key=lambda c: (c['teal_last'] - c['teal_first']) * np.sqrt(c['area']))
        if orig['teal_last'] - orig['teal_first'] > 0.25:
            rec['start_box'] = orig['box']
            rest = [c for c in comps if c is not orig]
            if rest:
                dst = max(rest, key=lambda c: c['area']); rec['end_box'] = dst['box']
    rec['blobs'] = comps
    out[ep] = rec
json.dump(out, open('diffgt.json', 'w'))
ok = sum('start_box' in v for v in out.values()); ok2 = sum('end_box' in v for v in out.values())
print(len(out), 'episodes; start box', ok, 'end box', ok2)
