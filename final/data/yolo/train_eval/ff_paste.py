"""Copy-paste training images for the weak classes: verified object crops (ff_targets.py) pasted onto team-scene
training images that already have complete labels (ds_v6, episodes >= 1000, train split, no repeat copies).

Every pasted object keeps its mask (feathered), is brightness-matched to the table where it lands, scaled a little with
depth, flipped at random, and never placed on an existing labelled box. Labels = the background's labels + the pasted
boxes. Episodes in HOLDOUT (every third new episode) contribute nothing, so they stay a clean test set.
usage: python -I ff_paste.py CROPS_DIR DS_V6 OUT N [--all]     (--all: no hold-out, for the final model)
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

WEAK = {'starfruit', 'green_chili', 'red_delicious', 'papaya', 'kiwi', 'passion_fruit', 'persimmon', 'orange', 'tomato',
        'mangosteen', 'carrot', 'brush_small', 'highlighter', 'lotion_bottle', 'contour'}


def holdout(uids):
    u = sorted(uids)
    return set(u[2::3])


def main():
    crops_dir, ds, out, n = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    use_all = '--all' in sys.argv
    rng = random.Random(7); nrng = np.random.default_rng(7)
    names = {}
    for line in (ds / 'dataset.yaml').read_text().splitlines():
        line = line.strip()
        if ':' in line and line.split(':')[0].strip().isdigit():
            names[line.split(':', 1)[1].strip()] = int(line.split(':')[0])
    meta = json.loads((crops_dir / 'crops.json').read_text())
    ho = set() if use_all else holdout({m['uid'] for m in meta})
    pool = [m for m in meta if m['uid'] not in ho]
    for m in pool:
        m['rgba'] = cv2.imread(str(crops_dir / 'crops' / f"{m['k']:03d}.png"), cv2.IMREAD_UNCHANGED)   # BGRA
    by_cls = {}
    for m in pool:
        by_cls.setdefault(m['cls'], []).append(m)
    classes = sorted(by_cls); w = [2.0 if c in WEAK else 1.0 for c in classes]
    bgs = sorted(p for p in (ds / 'images/train').glob('ep1*.jpg') if '_r' not in p.stem and int(p.stem.split('_')[0][2:]) >= 1000)
    (out / 'images/train').mkdir(parents=True, exist_ok=True); (out / 'labels/train').mkdir(parents=True, exist_ok=True)
    depth = lambda y: 0.75 + 0.5 * min(max((y - 200) / 280, 0), 1)
    made, used = 0, {}
    while made < n:
        bg = rng.choice(bgs)
        img = cv2.imread(str(bg)); H, W = img.shape[:2]
        lab_f = ds / 'labels/train' / (bg.stem + '.txt')
        rows = [l.split() for l in lab_f.read_text().splitlines() if l.strip()] if lab_f.exists() else []
        boxes = [[(float(r[1]) - float(r[3]) / 2) * W, (float(r[2]) - float(r[4]) / 2) * H, (float(r[1]) + float(r[3]) / 2) * W, (float(r[2]) + float(r[4]) / 2) * H] for r in rows]
        # table colour of this background (lower half, away from labelled boxes)
        free = np.ones((H, W), bool); free[:int(H * 0.5)] = False
        for b in boxes:
            free[max(int(b[1]) - 4, 0):int(b[3]) + 4, max(int(b[0]) - 4, 0):int(b[2]) + 4] = False
        if free.sum() < 2000:
            continue
        tab_bg = np.median(img[free], axis=0)   # BGR
        new_rows, placed = [], 0
        for _ in range(rng.choice([1, 2, 2, 3])):
            m = rng.choice(by_cls[rng.choices(classes, w)[0]])
            c = m['rgba']; ch, cw = c.shape[:2]
            for _try in range(30):
                cy_new = rng.uniform(H * 0.45, H - 8)
                s = rng.uniform(0.85, 1.15) * depth(cy_new) / depth((m['box'][1] + m['box'][3]) / 2)
                nw, nh = max(int(round(cw * s)), 6), max(int(round(ch * s)), 6)
                cx_new = rng.uniform(nw / 2 + 2, W - nw / 2 - 2)
                x0, y0 = int(cx_new - nw / 2), int(cy_new - nh / 2)
                if y0 < H * 0.36 or y0 + nh > H:
                    continue
                nb = [x0, y0, x0 + nw, y0 + nh]
                if any(max(0, min(nb[2], b[2] + 3) - max(nb[0], b[0] - 3)) * max(0, min(nb[3], b[3] + 3) - max(nb[1], b[1] - 3)) > 0 for b in boxes):
                    continue
                break
            else:
                continue
            c2 = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            if rng.random() < 0.5:
                c2 = c2[:, ::-1]
            gain = np.clip(tab_bg / np.maximum(np.array(m['table_rgb'][::-1]), 1.0), 0.8, 1.25)   # match the table brightness
            fgpix = np.clip(c2[..., :3].astype(np.float32) * gain, 0, 255)
            a = c2[..., 3].astype(np.float32) / 255
            if m.get('rect'):   # whole rectangle (thin brush): fade its own table into the new table at the border
                a = np.zeros((nh, nw), np.float32); e = max(min(nh, nw) // 5, 2); a[e:nh - e, e:nw - e] = 1
                a = cv2.GaussianBlur(a, (0, 0), e / 1.5)
                a = np.maximum(a, (np.linalg.norm(fgpix - tab_bg, axis=2) > 38).astype(np.float32))   # the brush itself stays solid
            a = cv2.GaussianBlur(a, (0, 0), 0.8)[..., None]
            roi = img[y0:y0 + nh, x0:x0 + nw].astype(np.float32)
            img[y0:y0 + nh, x0:x0 + nw] = np.clip(a * fgpix + (1 - a) * roi, 0, 255).astype(np.uint8)
            boxes.append(nb); placed += 1; used[m['cls']] = used.get(m['cls'], 0) + 1
            new_rows.append(f"{names[m['cls']]} {(nb[0] + nb[2]) / 2 / W:.6f} {(nb[1] + nb[3]) / 2 / H:.6f} {nw / W:.6f} {nh / H:.6f}")
        if not placed:
            continue
        stem = f'paste_{made:05d}'
        cv2.imwrite(str(out / 'images/train' / f'{stem}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / 'labels/train' / f'{stem}.txt').write_text('\n'.join([' '.join(r) for r in rows] + new_rows) + '\n')
        made += 1
    (out / 'paste_info.json').write_text(json.dumps({'images': made, 'backgrounds': len(bgs), 'crops_used': len(pool), 'holdout': sorted(ho),
                                                     'pasted_per_class': dict(sorted(used.items()))}, indent=1))
    print(json.dumps({'images': made, 'backgrounds': len(bgs), 'crops': len(pool), 'holdout_episodes': len(ho), 'pasted': dict(sorted(used.items()))}))


if __name__ == '__main__':
    main()
