"""Clean the team-scene part of a build_yolo dataset: in those images many objects carry no label (YOLO then learns them
as background) and some non-target labels are wrong (thin brush labelled brush_large). Per image keep only labels we can
trust -- the episode's instruction target, plates, organizer levels, a short list of classes that are not confused --
and paint every other non-table blob on the table with the table colour, so nothing visible is unlabelled.

usage: python -I ff_clean.py DS_V6 MERGED_LABELS.jsonl OUT [train|val]
"""
import collections
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path.home() / 'yolocue'))
import ff_trigger as FT  # noqa: E402

SAFE = {'eyeshadow_palette', 'powder_puff', 'lipstick', 'glitter_compact', 'pineapple', 'dragon_fruit', 'watermelon', 'grapes',
        'eggplant', 'green_apple', 'bell_pepper', 'strawberry', 'banana', 'pumpkin', 'mushroom', 'onion', 'lemon', 'avocado',
        'peach', 'cherry'}
G = {}


def init(ds, labels, out, split):
    names = {}
    for line in (Path(ds) / 'dataset.yaml').read_text().splitlines():
        line = line.strip()
        if ':' in line and line.split(':')[0].strip().isdigit():
            names[int(line.split(':')[0])] = line.split(':', 1)[1].strip()
    tgt = {}
    for l in open(labels):
        r = json.loads(l); L = (r.get('labels') or [{}])[0]
        pr, _ = FT.parse_row({'instruction': L.get('task_instruction'), 'custom': {c.get('name'): c.get('value') for c in (L.get('custom_fields') or [])}})
        tgt[r['episode_index']] = pr[1] if pr else None
    G.update(ds=Path(ds), out=Path(out), split=split, names=names, tgt=tgt)


def work(stem):
    ds, out, split, names = G['ds'], G['out'], G['split'], G['names']
    img = cv2.imread(str(ds / 'images' / split / f'{stem}.jpg')); H, W = img.shape[:2]
    rows = [l.split() for l in (ds / 'labels' / split / f'{stem}.txt').read_text().splitlines() if l.strip()]
    target = G['tgt'].get(int(stem.split('_')[0][2:]))
    keep, drop = [], []
    for r in rows:
        n = names[int(r[0])]
        (keep if n.startswith(('plate_', 'organizer_')) or n == target or n in SAFE else drop).append(r)
    box = lambda r, pad: [max(int((float(r[1]) - float(r[3]) / 2) * W) - pad, 0), max(int((float(r[2]) - float(r[4]) / 2) * H) - pad, 0),
                          min(int((float(r[1]) + float(r[3]) / 2) * W) + pad, W), min(int((float(r[2]) + float(r[4]) / 2) * H) + pad, H)]
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    tab = np.median(lab[int(H * 0.55):].reshape(-1, 3), axis=0)
    table = (np.linalg.norm(lab[..., 1:] - tab[1:], axis=2) < 10) & (np.abs(lab[..., 0] - tab[0]) < 50)
    frac = table.mean(1); top = next((y for y in range(H // 5, H - 20) if frac[y:y + 12].mean() > 0.55), int(H * 0.4))
    erase = (~table); erase[:top + 6] = False
    erase = cv2.dilate(cv2.morphologyEx(erase.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), np.ones((7, 7), np.uint8)).astype(bool)
    erase[:top + 2] = False
    for r in drop:
        b = box(r, 5); erase[max(b[1], top + 2):b[3], b[0]:b[2]] = True
    lv = [box(r, 0) for r in keep if names[int(r[0])].startswith('organizer_')]
    if lv:   # the acrylic organizer is one object: protect its whole extent, not only the three level boxes
        x0, y0, x1, y1 = min(b[0] for b in lv), min(b[1] for b in lv), max(b[2] for b in lv), max(b[3] for b in lv)
        px, py = int((x1 - x0) * 0.14), int((y1 - y0) * 0.12)
        erase[max(y0 - py, 0):y1 + py, max(x0 - px, 0):x1 + px] = False
    for r in keep:
        b = box(r, 5); erase[b[1]:b[3], b[0]:b[2]] = False
    n_erased = int(erase.sum())
    if n_erased:
        tabpix = img[int(H * 0.55):][table[int(H * 0.55):]]
        col = np.median(tabpix, axis=0) if len(tabpix) else np.array([100, 110, 40])
        fill = np.clip(col + np.random.default_rng(abs(hash(stem)) % 2 ** 32).normal(0, 2.0, img.shape), 0, 255)
        a = cv2.GaussianBlur(erase.astype(np.float32), (0, 0), 2.0)[..., None]
        img = np.clip(a * fill + (1 - a) * img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out / 'images' / split / f'{stem}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (out / 'labels' / split / f'{stem}.txt').write_text('\n'.join(' '.join(r) for r in keep) + ('\n' if keep else ''))
    return len(keep), [names[int(r[0])] for r in drop], n_erased > 400


def main():
    ds, labels, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]); split = sys.argv[4] if len(sys.argv) > 4 else 'train'
    (out / 'images' / split).mkdir(parents=True, exist_ok=True); (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
    (out / 'dataset.yaml').write_text((ds / 'dataset.yaml').read_text())
    stems = sorted(p.stem for p in (ds / 'images' / split).glob('ep1*.jpg') if '_r' not in p.stem and int(p.stem.split('_')[0][2:]) >= 1000)
    with Pool(6, initializer=init, initargs=(str(ds), labels, str(out), split)) as pool:
        res = pool.map(work, stems, chunksize=40)
    dropped = collections.Counter(n for _, d, _ in res for n in d)
    print(json.dumps({'images': len(res), 'labels_kept': sum(k for k, _, _ in res), 'labels_dropped': sum(dropped.values()),
                      'dropped_by_class': dict(dropped.most_common(14)), 'images_with_erased_area': sum(e for _, _, e in res)}))


if __name__ == '__main__':
    main()
