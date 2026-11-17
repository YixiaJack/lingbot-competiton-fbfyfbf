"""The new team first frames as whole training images. Labels = the objects verified by eye (ff_targets crops.json) +
plates and organizer levels from YOLO v5 (>= 0.5); everything else that is not table-coloured is painted with the table
colour, so no unverified object is left as unlabelled background. Held-out episodes (every third) are skipped unless --all.
usage: python -I new_whole.py FIRSTFRAMES TARGETS DS_V6 OUT [--all]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path.home() / 'yolocue'))
import yolocue_bridge as YC  # noqa: E402
from ff_paste import holdout  # noqa: E402


def main():
    ff, tg, ds, out = (Path(x) for x in sys.argv[1:5]); use_all = '--all' in sys.argv
    nid = {}
    for line in (ds / 'dataset.yaml').read_text().splitlines():
        line = line.strip()
        if ':' in line and line.split(':')[0].strip().isdigit():
            nid[line.split(':', 1)[1].strip()] = int(line.split(':')[0])
    meta = json.loads((tg / 'crops.json').read_text())
    uids = sorted({m['uid'] for m in meta}); ho = set() if use_all else holdout(uids)
    det = YC.Detector(YC.ROOT / 'weights/v5.torchscript', json.loads((YC.ROOT / 'weights/v5.names.json').read_text())['names'], conf=0.5)
    (out / 'images/train').mkdir(parents=True, exist_ok=True); (out / 'labels/train').mkdir(parents=True, exist_ok=True)
    n = nl = 0
    for uid in uids:
        if uid in ho:
            continue
        img = cv2.imread(str(ff / 'frames' / f'{uid}_first.jpg')); H, W = img.shape[:2]
        labels = [(m['cls'], [float(v) for v in m['box']]) for m in meta if m['uid'] == uid]
        labels += [(d[0], d[1:5]) for d in det(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if d[0].startswith(('plate_', 'organizer_'))]
        lab = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        tab = np.median(lab[int(H * 0.55):].reshape(-1, 3), axis=0)
        table = (np.linalg.norm(lab[..., 1:] - tab[1:], axis=2) < 10) & (np.abs(lab[..., 0] - tab[0]) < 50)
        frac = table.mean(1); top = next((y for y in range(H // 5, H - 20) if frac[y:y + 12].mean() > 0.55), int(H * 0.4))
        erase = cv2.dilate(cv2.morphologyEx((~table).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), np.ones((7, 7), np.uint8)).astype(bool)
        erase[:top + 6] = False
        lv = [b for c, b in labels if c.startswith('organizer_')]
        if lv:
            x0, y0, x1, y1 = min(b[0] for b in lv), min(b[1] for b in lv), max(b[2] for b in lv), max(b[3] for b in lv)
            px, py = int((x1 - x0) * 0.14), int((y1 - y0) * 0.12)
            erase[max(int(y0) - py, 0):int(y1) + py, max(int(x0) - px, 0):int(x1) + px] = False
        for c, b in labels:
            erase[max(int(b[1]) - 5, 0):int(b[3]) + 5, max(int(b[0]) - 5, 0):int(b[2]) + 5] = False
        if erase.any():
            tabpix = img[int(H * 0.55):][table[int(H * 0.55):]]
            col = np.median(tabpix, axis=0) if len(tabpix) else np.array([100, 110, 40])
            fill = np.clip(col + np.random.default_rng(abs(hash(uid)) % 2 ** 32).normal(0, 2.0, img.shape), 0, 255)
            a = cv2.GaussianBlur(erase.astype(np.float32), (0, 0), 2.0)[..., None]
            img = np.clip(a * fill + (1 - a) * img, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out / 'images/train' / f'new_{uid}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        (out / 'labels/train' / f'new_{uid}.txt').write_text(''.join(
            f"{nid[c]} {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}\n" for c, b in labels))
        n += 1; nl += len(labels)
    (out / 'dataset.yaml').write_text((ds / 'dataset.yaml').read_text())
    print(json.dumps({'images': n, 'labels': nl, 'held_out': len(ho)}))


if __name__ == '__main__':
    main()
