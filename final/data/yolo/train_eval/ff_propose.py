"""Label proposals for the first frames of the new team task-1/2 episodes (the ones YOLO v5/v6 never saw).

Per frame: (1) YOLO v5 detections >= 0.25 (cross-class NMS); (2) the target: class from the instruction / custom field,
position from the first/last frame difference (real_check diffgt) -- the YOLO box on that blob is relabelled, or the blob
itself becomes the box; (3) table-colour segmentation finds objects YOLO has no box for (class unknown). Every proposal
gets an id; review sheets (2x2 frames) are written for a visual pass that fixes classes and drops junk.
usage: python -I ff_propose.py DIR OUT   (DIR = firstframes dir with index.json + frames/)
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


def inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    i = inter(a, b); return i / (area(a) + area(b) - i + 1e-9)


def blobs(rgb, known=()):
    """Boxes of things on the table that are not table-coloured and not inside a known box (objects YOLO missed)."""
    lab = cv2.cvtColor(cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    H, W = lab.shape[:2]
    tab = np.median(lab[int(H * 0.55):].reshape(-1, 3), axis=0)
    d_ab = np.linalg.norm(lab[..., 1:] - tab[1:], axis=2); d_l = np.abs(lab[..., 0] - tab[0])
    table = (d_ab < 11) & (d_l < 55)
    rows = table.mean(1); top = next((y for y in range(H // 5, H - 20) if rows[y:y + 12].mean() > 0.55), int(H * 0.4))
    fg = (~table).astype(np.uint8); fg[:top + 4] = 0
    for b in known:   # known objects must not glue unknown neighbours into one blob
        fg[max(int(b[1]) - 3, 0):int(b[3]) + 3, max(int(b[0]) - 3, 0):int(b[2]) + 3] = 0
    k = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=2); fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=3)
    n, _, st, _ = cv2.connectedComponentsWithStats(fg)
    out = []
    for i in range(1, n):
        x, y, w, h, a = [int(v) for v in st[i]]
        if 220 <= a <= 30000 and w >= 12 and h >= 12 and a / (w * h) > 0.3 and max(w, h) / min(w, h) < 6 and not (y < top + 25 and h < 22):
            out.append([x, y, x + w, y + h])
    return out, top


def main():
    d, out = Path(sys.argv[1]), Path(sys.argv[2])
    (out / 'sheets').mkdir(parents=True, exist_ok=True)
    rows = json.loads((d / 'index.json').read_text())
    det = YC.detector(YC.load_config())
    F = ImageFont.load_default(size=13); FB = ImageFont.load_default(size=14)
    props, cells = {}, []
    todo = [r for r in rows if r.get('yolo_snapshot_ep') is None and FT.parse_row(r)[0]]
    for n, r in enumerate(todo, 1):
        pr, how = FT.parse_row(r)
        f0 = d / 'frames' / f"{r['uid']}_first.jpg"
        rgb = cv2.cvtColor(cv2.imread(str(f0)), cv2.COLOR_BGR2RGB)
        alld = det(rgb)
        dets = [x for x in alld if x[5] >= 0.25]
        keep = []
        for x in sorted(dets, key=lambda x: -x[5]):
            if all(iou(x[1:5], k[1:5]) < 0.6 for k in keep):
                keep.append(x)
        P = [{'box': x[1:5], 'src': 'yolo', 'cls': x[0], 'conf': x[5], 'target': False} for x in keep]
        low = []
        for x in sorted((x for x in alld if 0.05 <= x[5] < 0.25 and not x[0].startswith(('plate_', 'organizer_'))), key=lambda x: -x[5]):
            if all(iou(x[1:5], k['box']) < 0.3 and inter(x[1:5], k['box']) / max(area(x[1:5]), 1) < 0.5 for k in P + low):
                low.append({'box': x[1:5], 'src': 'yolo-low', 'cls': None, 'hint': x[0], 'conf': x[5], 'target': False})
        P += low
        g = FT.diffgt(f0, d / 'frames' / f"{r['uid']}_last.jpg") if r.get('result') in ('成功', '部分成功') else None
        if g and (area(g) > 15000 or any(p['cls'] and p['cls'].startswith('plate_') and inter(g, p['box']) / area(g) > 0.25 for p in P)):
            g = None   # blob glued to the plate / to neighbours: not a box, the target is picked by eye
        tgt = None
        if g:
            cand = [p for p in P if not (p['cls'] or '').startswith(('plate_', 'organizer_')) and inter(p['box'], g) / max(min(area(p['box']), area(g)), 1) > 0.5]
            if cand:
                tgt = max(cand, key=lambda p: inter(p['box'], g))
                tgt['yolo_cls'] = tgt['cls'] or tgt.get('hint'); tgt['cls'] = pr[1]; tgt['target'] = True; tgt['src'] += '+diff'
            else:
                tgt = {'box': [float(v) for v in g], 'src': 'diff', 'cls': pr[1], 'conf': None, 'target': True}
                P.append(tgt)
        bl, top = blobs(rgb, [p['box'] for p in P])
        for b in bl:
            if all(inter(b, p['box']) / max(area(b), 1) < 0.5 and inter(b, p['box']) / max(area(p['box']), 1) < 0.5 for p in P):
                P.append({'box': [float(v) for v in b], 'src': 'blob', 'cls': None, 'conf': None, 'target': False})
        for i, p in enumerate(P):
            p['id'] = i
        props[r['uid']] = {'n': n, 'instruction': r['instruction'], 'result': r.get('result'), 'target_cls': pr[1], 'dest_cls': pr[2],
                           'target_found': tgt is not None, 'props': P}
        im = Image.fromarray(rgb); dr = ImageDraw.Draw(im)
        for p in P:
            col = (255, 255, 255) if p['target'] else (255, 60, 60) if p['src'] == 'blob' else (255, 150, 220) if p['src'] == 'yolo-low' else (255, 230, 0)
            if p['cls'] and p['cls'].startswith(('plate_', 'organizer_')):
                col = (120, 200, 255)
            b = p['box']; dr.rectangle(b, outline=col, width=2)
            lab = f"{p['id']}:{p['cls'] or '?' + (p.get('hint') or '')}" + (f" {p['conf']:.2f}" if p['conf'] else '') + (' *T*' if p['target'] else '')
            tw = dr.textlength(lab, font=F); ty = b[1] - 15 if b[1] >= 15 else b[3] + 1
            dr.rectangle([b[0], ty, b[0] + tw + 4, ty + 14], fill=col); dr.text((b[0] + 2, ty), lab, font=F, fill=(0, 0, 0))
        cell = Image.new('RGB', (640, 480 + 40), (255, 255, 255)); cell.paste(im, (0, 40)); g2 = ImageDraw.Draw(cell)
        g2.text((4, 2), f"[{n}] {r['uid']}  target={pr[1]} dest={pr[2]}  ({r.get('result')})" + ('' if tgt else '   TARGET NOT LOCATED'), font=FB, fill=(0, 0, 0) if tgt else (200, 0, 0))
        g2.text((4, 21), (r['instruction'] or '')[:95], font=FB, fill=(0, 0, 160))
        cells.append(cell); cell.save(out / f"{n:02d}_{r['uid']}.jpg", quality=90)
    for k in range(0, len(cells), 4):
        c = cells[k:k + 4]; sheet = Image.new('RGB', (1280, ((len(c) + 1) // 2) * 520), (255, 255, 255))
        for i, im in enumerate(c):
            sheet.paste(im, ((i % 2) * 640, (i // 2) * 520))
        sheet.save(out / 'sheets' / f'p{k // 4 + 1:02d}.jpg', quality=85)
    (out / 'proposals.json').write_text(json.dumps(props, ensure_ascii=False, indent=1))
    print(json.dumps({'frames': len(todo), 'target_located': sum(v['target_found'] for v in props.values()),
                      'proposals': sum(len(v['props']) for v in props.values()),
                      'blob_only': sum(p['src'] == 'blob' for v in props.values() for p in v['props']), 'sheets': (len(cells) + 3) // 4}))


if __name__ == '__main__':
    main()
