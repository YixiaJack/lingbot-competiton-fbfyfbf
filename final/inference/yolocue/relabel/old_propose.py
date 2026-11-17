"""Frame 0 of every team episode in a build_yolo dataset (ds_v6, episodes >= 1000, train + val) with what is known about
each object, numbered for a visual pass that fixes the labels:
  old   an existing label (LocateAnything pipeline) -- class may be wrong
  yolo  a YOLO v5 detection (>= 0.25) where no label exists
  low   a low-confidence YOLO box (0.05-0.25), class unknown
  blob  a non-table blob nothing else covers, class unknown
usage: python -I old_propose.py DS_V6 MERGED_LABELS.jsonl OUT
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path.home() / 'yolocue'))
import ff_trigger as FT  # noqa: E402
import yolocue_bridge as YC  # noqa: E402

area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
inter = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
PL = ('plate_', 'organizer_')


def covered(b, P):
    return any(inter(b, p['box']) / max(min(area(b), area(p['box'])), 1) > 0.5 for p in P)


def main():
    ds, out = Path(sys.argv[1]), Path(sys.argv[3]); (out / 'sheets').mkdir(parents=True, exist_ok=True)
    names = {}
    for line in (ds / 'dataset.yaml').read_text().splitlines():
        line = line.strip()
        if ':' in line and line.split(':')[0].strip().isdigit():
            names[int(line.split(':')[0])] = line.split(':', 1)[1].strip()
    ins = {}
    for l in open(sys.argv[2]):
        r = json.loads(l); L = (r.get('labels') or [{}])[0]
        ins[r['episode_index']] = {'instruction': L.get('task_instruction'), 'result': L.get('result'),
                                   'custom': {c.get('name'): c.get('value') for c in (L.get('custom_fields') or [])}}
    det = YC.Detector(YC.ROOT / 'weights/v5.torchscript', json.loads((YC.ROOT / 'weights/v5.names.json').read_text())['names'], conf=0.01)
    F = ImageFont.load_default(size=13); FB = ImageFont.load_default(size=14)
    eps = {}
    for split in ('train', 'val'):
        for p in (ds / 'images' / split).glob('ep1*.jpg'):
            e, fr = p.stem.split('_')[0], p.stem.split('_')[1]
            if '_r' in p.stem or int(e[2:]) < 1000:
                continue
            eps.setdefault((int(e[2:]), split), []).append(int(fr))
    props, cells = {}, []
    for n, ((e, split), frames) in enumerate(sorted(eps.items()), 1):
        f0 = min(frames); stem = f'ep{e}_{f0:04d}'
        rgb = cv2.cvtColor(cv2.imread(str(ds / 'images' / split / f'{stem}.jpg')), cv2.COLOR_BGR2RGB); H, W = rgb.shape[:2]
        P = []
        for l in (ds / 'labels' / split / f'{stem}.txt').read_text().splitlines():
            r = l.split()
            if r:
                x, y, w, h = [float(v) for v in r[1:]]
                P.append({'box': [round((x - w / 2) * W, 1), round((y - h / 2) * H, 1), round((x + w / 2) * W, 1), round((y + h / 2) * H, 1)], 'src': 'old', 'cls': names[int(r[0])], 'conf': None})
        dets = det(rgb)
        for x in sorted(dets, key=lambda x: -x[5]):
            if x[5] >= 0.05 and not x[0].startswith(PL) and area(x[1:5]) < 45000 and not covered(x[1:5], P):
                P.append({'box': x[1:5], 'src': 'yolo' if x[5] >= 0.25 else 'low', 'cls': x[0], 'conf': x[5]})
        lab = cv2.cvtColor(cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        tab = np.median(lab[int(H * 0.55):].reshape(-1, 3), axis=0)
        table = (np.linalg.norm(lab[..., 1:] - tab[1:], axis=2) < 11) & (np.abs(lab[..., 0] - tab[0]) < 55)
        frac = table.mean(1); top = next((y for y in range(H // 5, H - 20) if frac[y:y + 12].mean() > 0.55), int(H * 0.4))
        fg = (~table).astype(np.uint8); fg[:top + 6] = 0
        for p in P:
            b = p['box']
            if p['cls'].startswith('organizer_'):
                continue
            fg[max(int(b[1]) - 4, 0):int(b[3]) + 4, max(int(b[0]) - 4, 0):int(b[2]) + 4] = 0
        lv = [p['box'] for p in P if p['cls'].startswith('organizer_')]
        if lv:   # no blob proposals inside the acrylic organizer
            x0, y0, x1, y1 = min(b[0] for b in lv), min(b[1] for b in lv), max(b[2] for b in lv), max(b[3] for b in lv)
            px, py = int((x1 - x0) * 0.16), int((y1 - y0) * 0.12)
            fg[max(int(y0) - py, 0):int(y1) + py, max(int(x0) - px, 0):int(x1) + px] = 0
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        k, _, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=3))
        for i in range(1, k):
            x, y, w, h, a = [int(v) for v in st[i]]
            if 350 <= a <= 30000 and min(w, h) >= 14 and max(w, h) / min(w, h) < 7 and a / (w * h) > 0.25:
                P.append({'box': [x, y, x + w, y + h], 'src': 'blob', 'cls': None, 'conf': None})
        for i, p in enumerate(P):
            p['id'] = i
        I = ins.get(e, {}); pr, _ = FT.parse_row(I) if I else (None, None)
        props[str(e)] = {'n': n, 'split': split, 'frame0': f0, 'frames': sorted(frames), 'instruction': I.get('instruction'), 'result': I.get('result'),
                         'target_cls': pr[1] if pr else None, 'dest_cls': pr[2] if pr else None, 'props': P}
        im = Image.fromarray(rgb); dr = ImageDraw.Draw(im)
        for p in P:
            col = {'old': (255, 235, 0), 'yolo': (0, 255, 120), 'low': (255, 150, 220), 'blob': (255, 60, 60)}[p['src']]
            if p['cls'] and p['cls'].startswith(PL):
                col = (120, 200, 255)
            b = p['box']; dr.rectangle(b, outline=col, width=2)
            t = f"{p['id']}:{p['cls'] or '?'}" + (f" {p['conf']:.2f}" if p['conf'] else '')
            tw = dr.textlength(t, font=F); ty = b[1] - 15 if b[1] >= 15 else b[3] + 1
            dr.rectangle([b[0], ty, b[0] + tw + 4, ty + 14], fill=col); dr.text((b[0] + 2, ty), t, font=F, fill=(0, 0, 0))
        cell = Image.new('RGB', (640, 520), (255, 255, 255)); cell.paste(im, (0, 40)); g = ImageDraw.Draw(cell)
        g.text((4, 2), f"[{n}] ep{e} ({split}, {len(frames)} frames)  target={pr[1] if pr else '?'} dest={pr[2] if pr else '?'}", font=FB, fill=(0, 0, 0))
        g.text((4, 21), (I.get('instruction') or '')[:95], font=FB, fill=(0, 0, 160))
        cells.append(cell)
    for k0 in range(0, len(cells), 4):
        c = cells[k0:k0 + 4]; sheet = Image.new('RGB', (1280, ((len(c) + 1) // 2) * 520), (255, 255, 255))
        for i, im in enumerate(c):
            sheet.paste(im, ((i % 2) * 640, (i // 2) * 520))
        sheet.save(out / 'sheets' / f'o{k0 // 4 + 1:02d}.jpg', quality=85)
    (out / 'old_proposals.json').write_text(json.dumps(props, ensure_ascii=False, indent=1))
    import collections
    print(json.dumps({'episodes': len(props), 'by_split': dict(collections.Counter(v['split'] for v in props.values())),
                      'props': dict(collections.Counter(p['src'] for v in props.values() for p in v['props'])), 'sheets': (len(cells) + 3) // 4}))


if __name__ == '__main__':
    main()
