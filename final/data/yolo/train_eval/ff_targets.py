"""Tight box + mask of labelled objects in the new team first frames, for copy-paste training data.

Target of every new task-1/2 episode: class from the instruction (ff_trigger.parse_row), position from, in order,
  yolo-class   the one YOLO v5 box of that class (conf >= 0.05) that the first/last frame difference agrees with (or
               the only one when there is no clean difference blob);
  yolo+diff    any YOLO box (conf >= 0.01, not a plate/level) sitting on a clean difference blob;
  diff         the clean difference blob itself, tightened to its non-table pixels;
  seed         a hand-given point or box (seeds.json), tightened the same way.
seeds.json also adds other objects seen in the frames ({uid: {"target": {...}|"drop", "more": [{"cls":..,"pt":[x,y]}]}}).
Mask = non-table-coloured pixels inside the box (largest blob, closed, feathered when pasted).
Outputs OUT/crops/<k>.png (RGBA), OUT/crops.json, OUT/gallery_NN.jpg (crop + class, for a visual check).
usage: python -I ff_targets.py DIR OUT [seeds.json]
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
PLATE = ('plate_', 'organizer_')


def table_model(rgb):
    lab = cv2.cvtColor(cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    tab = np.median(lab[int(lab.shape[0] * 0.55):].reshape(-1, 3), axis=0)
    fg = (np.linalg.norm(lab[..., 1:] - tab[1:], axis=2) >= 10) | (np.abs(lab[..., 0] - tab[0]) >= 50)
    return fg.astype(np.uint8)


def tighten(fg, box, pt=None, pad=2):
    """Largest non-table blob inside box (the one under pt when given) -> (tight box, mask in tight-box coords)."""
    H, W = fg.shape
    x0, y0, x1, y1 = max(int(box[0]) - pad, 0), max(int(box[1]) - pad, 0), min(int(box[2]) + pad, W), min(int(box[3]) + pad, H)
    sub = cv2.morphologyEx(fg[y0:y1, x0:x1].copy(), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sub)
    if n <= 1:
        return None
    # the blob under the hand-given point, else the one under the box centre (plates and neighbours that only clip
    # the box corners are other blobs), else the largest
    cands = ([pt] if pt is not None else []) + [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)]
    i = 0
    for q in cands:
        qy, qx = int(q[1]) - y0, int(q[0]) - x0
        if 0 <= qy < sub.shape[0] and 0 <= qx < sub.shape[1] and lab[qy, qx] > 0 and st[lab[qy, qx], 4] >= 0.25 * st[1:, 4].max():
            i = int(lab[qy, qx]); break
    if i == 0:
        i = 1 + int(np.argmax(st[1:, 4]))
    x, y, w, h, a = [int(v) for v in st[i]]
    if a < 60:
        return None
    m = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); m2 = np.zeros_like(m); cv2.drawContours(m2, cs, -1, 1, -1)
    return [x0 + x, y0 + y, x0 + x + w, y0 + y + h], m2


def from_point(fg, pt, dets, plates, raw=False):
    """Object under a hand-given point: smallest YOLO box around it, else the colour blob there."""
    c = [d for d in dets if not d[0].startswith(PLATE) and d[5] >= 0.05 and d[1] <= pt[0] <= d[3] and d[2] <= pt[1] <= d[4] and area(d[1:5]) < 40000]
    if c:
        b = max(c, key=lambda d: d[5])[1:5]
        if raw:   # thin objects (brush handles) do not survive the colour mask: keep the detector's box
            b = [int(round(v)) for v in b]
            return b, np.ones((b[3] - b[1], b[2] - b[0]), np.uint8)
        return tighten(fg, b, pt, pad=2)
    f2 = fg.copy()
    for p in plates:
        if not (p[0] <= pt[0] <= p[2] and p[1] <= pt[1] <= p[3]):
            f2[int(p[1]):int(p[3]), int(p[0]):int(p[2])] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(f2, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))
    i = lab[int(pt[1]), int(pt[0])]
    if i == 0:
        return None
    x, y, w, h, a = [int(v) for v in st[i]]
    return tighten(fg, [x, y, x + w, y + h], pt, pad=0) if a < 30000 else None


def main():
    d, out = Path(sys.argv[1]), Path(sys.argv[2])
    seeds = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else {}
    (out / 'crops').mkdir(parents=True, exist_ok=True)
    rows = json.loads((d / 'index.json').read_text())
    det = YC.Detector(YC.ROOT / 'weights/v5.torchscript', json.loads((YC.ROOT / 'weights/v5.names.json').read_text())['names'], conf=0.01)
    items, missing = [], []
    for r in rows:
        pr, _ = FT.parse_row(r)
        if r.get('yolo_snapshot_ep') is not None or not pr:
            continue
        f0 = d / 'frames' / f"{r['uid']}_first.jpg"
        rgb = cv2.cvtColor(cv2.imread(str(f0)), cv2.COLOR_BGR2RGB)
        fg = table_model(rgb); dets = det(rgb)
        plates = [x[1:5] for x in dets if x[0].startswith('plate_') and x[5] >= 0.5]
        for b in plates:   # a plate that clips an object's box must not end up in the object's mask
            cv2.ellipse(fg, (int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)), (max(int((b[2] - b[0]) / 2) - 1, 1), max(int((b[3] - b[1]) / 2) - 1, 1)), 0, 0, 360, 0, -1)
        tabpix = rgb[int(rgb.shape[0] * 0.55):][fg[int(rgb.shape[0] * 0.55):] == 0]
        tab = [float(v) for v in np.median(tabpix, axis=0)] if len(tabpix) else [40.0, 110.0, 100.0]
        S = seeds.get(r['uid'], {})
        got, src = None, None
        if S.get('target') == 'drop':
            src = 'dropped by hand'
        elif isinstance(S.get('target'), dict):
            t = S['target']
            raw = t.get('raw') or pr[1] == 'brush_small'
            got = (None if raw else tighten(fg, t['box'], pad=0)) if 'box' in t else from_point(fg, t['pt'], dets, plates, raw); src = 'seed'
            if got is None and 'box' in t:
                got = ([int(v) for v in t['box']], np.ones((int(t['box'][3] - t['box'][1]), int(t['box'][2] - t['box'][0])), np.uint8)); src = 'seed box as given'
        else:
            g = FT.diffgt(f0, d / 'frames' / f"{r['uid']}_last.jpg") if r.get('result') in ('成功', '部分成功') else None
            if g and (area(g) > 15000 or any(inter(g, p) / area(g) > 0.25 for p in plates)):
                g = None
            own = [x for x in dets if x[0] == pr[1] and x[5] >= 0.05]
            if g:
                c = [x for x in own if inter(x[1:5], g) / max(min(area(x[1:5]), area(g)), 1) > 0.5]
                if c:
                    got, src = tighten(fg, max(c, key=lambda x: x[5])[1:5]), 'yolo-class+diff'
                else:
                    c = [x for x in dets if not x[0].startswith(PLATE) and inter(x[1:5], g) / max(min(area(x[1:5]), area(g)), 1) > 0.5 and 0.4 < area(x[1:5]) / area(g) < 2.5]
                    got, src = (tighten(fg, max(c, key=lambda x: x[5])[1:5]), 'yolo+diff') if c else (tighten(fg, g), 'diff')
            elif len(own) == 1 and own[0][5] >= 0.3:
                got, src = tighten(fg, own[0][1:5]), 'yolo-class'
        if got is None:
            if src != 'dropped by hand':
                missing.append(r['uid'])
        else:
            items.append({'uid': r['uid'], 'cls': pr[1], 'role': 'target', 'src': src, 'box': got[0], 'mask': got[1], 'rgb': rgb, 'tab': tab})
        for m in S.get('more', []):
            g2 = tighten(fg, m['box'], pad=0) if 'box' in m else from_point(fg, m['pt'], dets, plates, m['cls'] == 'brush_small')
            if g2 is None:
                missing.append(f"{r['uid']} more {m['cls']} {m.get('pt') or m.get('box')}")
            else:
                items.append({'uid': r['uid'], 'cls': m['cls'], 'role': 'other', 'src': 'seed', 'box': g2[0], 'mask': g2[1], 'rgb': rgb, 'tab': tab})
    F = ImageFont.load_default(size=13)
    meta, cells = [], []
    for k, it in enumerate(items):
        x0, y0, x1, y1 = it['box']; crop = it['rgb'][y0:y1, x0:x1]
        rgba = np.dstack([crop, (it['mask'] * 255).astype(np.uint8)])
        Image.fromarray(rgba).save(out / 'crops' / f'{k:03d}.png')
        meta.append({'k': k, 'uid': it['uid'], 'cls': it['cls'], 'role': it['role'], 'src': it['src'], 'box': it['box'], 'table_rgb': it['tab'],
                     'rect': bool(it['mask'].all())})
        # gallery cell: context (box drawn) on the left, masked crop on the right
        pad = 45; cx0, cy0, cx1, cy1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, 640), min(y1 + pad, 480)
        ctx = Image.fromarray(it['rgb'][cy0:cy1, cx0:cx1]).copy(); ImageDraw.Draw(ctx).rectangle([x0 - cx0, y0 - cy0, x1 - cx0 - 1, y1 - cy0 - 1], outline=(255, 255, 255), width=1)
        s = min(150 / ctx.width, 150 / ctx.height); ctx = ctx.resize((max(int(ctx.width * s), 1), max(int(ctx.height * s), 1)), Image.LANCZOS)
        bg = np.full((crop.shape[0], crop.shape[1], 3), 255, np.uint8); bg[it['mask'] > 0] = crop[it['mask'] > 0]
        cm = Image.fromarray(bg); s = min(150 / cm.width, 150 / cm.height); cm = cm.resize((max(int(cm.width * s), 1), max(int(cm.height * s), 1)), Image.LANCZOS)
        cell = Image.new('RGB', (320, 190), (225, 225, 225)); cell.paste(ctx, (4, 36)); cell.paste(cm, (164, 36)); g = ImageDraw.Draw(cell)
        g.text((4, 2), f"{k}: {it['cls']}  [{it['role']}]", font=F, fill=(0, 0, 0)); g.text((4, 18), f"{it['uid']}  {it['src']}", font=F, fill=(90, 90, 90))
        cells.append(cell)
    for s0 in range(0, len(cells), 24):
        c = cells[s0:s0 + 24]; sheet = Image.new('RGB', (4 * 320, ((len(c) + 3) // 4) * 190), (255, 255, 255))
        for i, im in enumerate(c):
            sheet.paste(im, ((i % 4) * 320, (i // 4) * 190))
        sheet.save(out / f'gallery_{s0 // 24 + 1:02d}.jpg', quality=88)
    (out / 'crops.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    import collections
    print(json.dumps({'crops': len(meta), 'targets': sum(m['role'] == 'target' for m in meta), 'by_src': dict(collections.Counter(m['src'] for m in meta)),
                      'missing': missing}, ensure_ascii=False))


if __name__ == '__main__':
    main()
