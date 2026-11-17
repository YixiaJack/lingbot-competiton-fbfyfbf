"""Apply the hand fixes (old_fix.json) to the team episodes of ds_v6 and write the corrected dataset part.

Per episode the frame-0 label set is rebuilt: existing labels and YOLO boxes keep their class unless the fix changes or
drops it; low-confidence boxes and colour blobs only become labels when the fix names their class; "add" gives extra
objects by point/box; "unknown" objects are painted with the table colour (never left as unlabelled background).
The scene does not change before the pick-up, so the same labels go onto the episode's frames up to MAXF.
usage: python -I old_apply.py DS_V6 PROPOSALS.json FIX.json OUT [MAXF=60]
"""
import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path.home() / 'yolocue'))
import ff_targets as T  # noqa: E402
import yolocue_bridge as YC  # noqa: E402


def main():
    ds, out = Path(sys.argv[1]), Path(sys.argv[4]); maxf = int(sys.argv[5]) if len(sys.argv) > 5 else 60
    props = json.loads(Path(sys.argv[2]).read_text()); fix = json.loads(Path(sys.argv[3]).read_text())
    nid = {}
    for line in (ds / 'dataset.yaml').read_text().splitlines():
        line = line.strip()
        if ':' in line and line.split(':')[0].strip().isdigit():
            nid[line.split(':', 1)[1].strip()] = int(line.split(':')[0])
    det = YC.Detector(YC.ROOT / 'weights/v5.torchscript', json.loads((YC.ROOT / 'weights/v5.names.json').read_text())['names'], conf=0.01)
    F = ImageFont.load_default(size=13)
    stats, cells, final = collections.Counter(), [], {}
    for e, v in sorted(props.items(), key=lambda kv: int(kv[0])):
        split, f0 = v['split'], v['frame0']; fx = fix.get(e, {})
        (out / 'images' / split).mkdir(parents=True, exist_ok=True); (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
        bgr0 = cv2.imread(str(ds / 'images' / split / f'ep{e}_{f0:04d}.jpg')); rgb = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB); H, W = rgb.shape[:2]
        fg = T.table_model(rgb); dets = det(rgb)
        plates = [p['box'] for p in v['props'] if (p['cls'] or '').startswith('plate_')]
        labels, erase = [], [list(b) for b in fx.get('unknown_boxes', [])]
        for p in v['props']:
            i = p['id']; cls = fx.get('set', {}).get(str(i))
            if i in fx.get('drop', []):
                stats['dropped'] += 1; continue
            if i in fx.get('unknown', []):
                erase.append(p['box']); stats['erased'] += 1; continue
            if cls is None:
                if p['src'] in ('low', 'blob'):
                    continue
                cls = p['cls']
            else:
                stats['class changed' if p['src'] in ('old', 'yolo') and p['cls'] != cls else 'labelled (was unlabelled)' if p['src'] in ('low', 'blob') else 'confirmed'] += 1
            box = p['box']
            if p['src'] == 'blob':   # tighten a colour blob to the object
                t = T.tighten(fg, box, pad=0); box = t[0] if t else box
            labels.append((cls, [float(x) for x in box]))
        for a in fx.get('add', []):
            if a.get('raw'):
                box = a['box']
            else:
                t = T.tighten(fg, a['box'], pad=0) if 'box' in a else T.from_point(fg, a['pt'], dets, plates, a['cls'] == 'brush_small')
                if t is None:
                    print('ADD FAILED', e, a); continue
                box = t[0]
            labels.append((a['cls'], [float(x) for x in box])); stats['added'] += 1
        final[e] = {'split': split, 'labels': labels, 'erase': erase}
        tabpix = bgr0[int(H * 0.55):][fg[int(H * 0.55):] == 0]
        col = np.median(tabpix, axis=0) if len(tabpix) else np.array([100, 110, 40])
        txt = ''.join(f"{nid[c]} {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}\n" for c, b in labels)
        for fr in v['frames']:
            if fr > maxf:
                continue
            img = bgr0 if fr == f0 else cv2.imread(str(ds / 'images' / split / f'ep{e}_{fr:04d}.jpg'))
            img = img.copy()
            if erase:
                m = np.zeros((H, W), np.float32)
                for b in erase:
                    m[max(int(b[1]) - 6, 0):int(b[3]) + 6, max(int(b[0]) - 6, 0):int(b[2]) + 6] = 1
                a = cv2.GaussianBlur(m, (0, 0), 2.0)[..., None]
                fill = np.clip(col + np.random.default_rng(int(e) * 1000 + fr).normal(0, 2.0, img.shape), 0, 255)
                img = np.clip(a * fill + (1 - a) * img, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out / 'images' / split / f'ep{e}_{fr:04d}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
            (out / 'labels' / split / f'ep{e}_{fr:04d}.txt').write_text(txt); stats[f'images {split}'] += 1
        # review cell: final labels on frame 0 (after erasing)
        im = Image.fromarray(cv2.cvtColor(cv2.imread(str(out / 'images' / split / f'ep{e}_{f0:04d}.jpg')), cv2.COLOR_BGR2RGB)); g = ImageDraw.Draw(im)
        for c, b in labels:
            colr = (120, 200, 255) if c.startswith(('plate_', 'organizer_')) else (255, 235, 0)
            g.rectangle(b, outline=colr, width=2); tw = g.textlength(c, font=F); ty = b[1] - 15 if b[1] >= 15 else b[3] + 1
            g.rectangle([b[0], ty, b[0] + tw + 4, ty + 14], fill=colr); g.text((b[0] + 2, ty), c, font=F, fill=(0, 0, 0))
        cell = Image.new('RGB', (640, 502), (255, 255, 255)); cell.paste(im, (0, 22)); ImageDraw.Draw(cell).text((4, 3), f"ep{e} ({split})  {v['instruction']}", font=F, fill=(0, 0, 160))
        cells.append(cell)
    (out / 'dataset.yaml').write_text((ds / 'dataset.yaml').read_text())
    (out / 'final_labels.json').write_text(json.dumps(final, indent=1))
    (out / 'sheets').mkdir(exist_ok=True)
    for k in range(0, len(cells), 6):
        c = cells[k:k + 6]; sheet = Image.new('RGB', (1920, ((len(c) + 2) // 3) * 502), (255, 255, 255))
        for i, im in enumerate(c):
            sheet.paste(im, ((i % 3) * 640, (i // 3) * 502))
        sheet.save(out / 'sheets' / f'f{k // 6 + 1:02d}.jpg', quality=85)
    stats['object labels'] = sum(1 for v in final.values() for c, _ in v['labels'] if not c.startswith(('plate_', 'organizer_')))
    print(json.dumps(dict(stats)))


if __name__ == '__main__':
    main()
