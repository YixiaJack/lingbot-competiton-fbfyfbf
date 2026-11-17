"""Builds the reference layouts for the board's /topref page (live top camera + ghosts of a past scene).
Run on the Mac with ~/lingbot_v7env/bin/python (cv2, numpy):
    ~/lingbot_v7env/bin/python mk_topref.py 20261115 200600 204100     (date, from HHMMSS, to HHMMSS)
For every task-1/2 session in that window (yolocue records on the 5080, fetched read-only over ssh):
  - topref/ref_<id>.jpg   the session's first top-camera frame (the yolocue _cue.jpg) with the white target box and the
                          orange destination box painted out (5 px PIL rectangles at known coordinates -> cv2.inpaint)
  - topref/obj_<id>.png   RGBA: only the objects (GrabCut inside each YOLO box of the start detection), soft edges
  - topref/edge_<id>.png  RGBA: 2 px outline of each object
  - topref/line_<id>.png  RGBA: edge lines of the whole frame (Canny) -- shows the transparent organizer too
  - topref/refs.json      list of references (time, task, target, destination, objects with centroid / box)
  - topref/check_<date>_<from>.jpg  contact sheet for a human check (frame with boxes | painted out | objects)
No box is drawn anywhere; the boxes only seed the object masks."""
import io, json, os, subprocess, sys, tarfile
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'topref')
HOST = os.environ.get('HOST', 'lbreal')
LOGS = 'workspace/lingbot/inference/logs/yolocue'
LINE, MIN_CONF = 5, 0.20
NO_MASK = ('organizer_top', 'organizer_middle', 'organizer_bottom')   # transparent: GrabCut cannot separate it


def fetch(date, t0, t1):
    """jsonl + _cue.jpg of the sessions in [t0, t1] as {stem: {'jsonl': text, 'cue': bytes}} (one read-only tar over ssh)."""
    ls = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', HOST, f'ls {LOGS}'], capture_output=True, text=True, timeout=60)
    names = []
    for n in ls.stdout.split():
        q = n.split('_')
        if len(q) >= 3 and q[0] == date and t0 <= q[1] <= t1 and (n.endswith('_cue.jpg') or (len(q) == 3 and n.endswith('.jsonl'))):
            names.append(n)
    if not names:
        return {}
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', HOST, f'cd {LOGS} && tar -cf - ' + ' '.join(names)],
                       capture_output=True, timeout=120)
    if r.returncode != 0:
        sys.exit('fetch failed: ' + r.stderr.decode()[-300:])
    got = {}
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tf:
        for m in tf.getmembers():
            stem = m.name[:-6] if m.name.endswith('.jsonl') else m.name[:-8]
            got.setdefault(stem, {})['jsonl' if m.name.endswith('.jsonl') else 'cue'] = tf.extractfile(m).read()
    return {k: v for k, v in got.items() if 'jsonl' in v and 'cue' in v}


def box_band(shape, box):
    """mask of the 5 px PIL outline of [x1, y1, x2-1, y2-1] (drawn inwards), widened for JPEG ringing: 2 px outwards,
    1 px inwards (the object touches the inner side)."""
    h, w = shape[:2]; m = np.zeros((h, w), np.uint8)
    x1, y1, x2, y2 = (int(round(v)) for v in box); x2 -= 1; y2 -= 1; g, i = 2, 1
    for (a, b, c, d) in ((x1 - g, y1 - g, x2 + g, y1 + LINE - 1 + i), (x1 - g, y2 - LINE + 1 - i, x2 + g, y2 + g),
                         (x1 - g, y1 - g, x1 + LINE - 1 + i, y2 + g), (x2 - LINE + 1 - i, y1 - g, x2 + g, y2 + g)):
        m[max(b, 0):min(d + 1, h), max(a, 0):min(c + 1, w)] = 255
    return m


def fill_holes(a):
    ff = a.copy(); h, w = a.shape; mk = np.zeros((h + 2, w + 2), np.uint8)
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if ff[seed[1], seed[0]] == 0:
            cv2.floodFill(ff, mk, seed, 255)
    return a | cv2.bitwise_not(ff)


def object_mask(img, box):
    """GrabCut inside the YOLO box (6 px context); largest component, holes filled; full-frame uint8 mask or None."""
    H, W = img.shape[:2]; x0, y0, x1, y1 = (int(round(v)) for v in box); m = 6
    X0, Y0, X1, Y1 = max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, W), min(y1 + m, H)
    sub = img[Y0:Y1, X0:X1].copy(); gm = np.zeros(sub.shape[:2], np.uint8)
    rect = (max(x0 - X0, 0), max(y0 - Y0, 0), max(min(x1, W) - max(x0, 0), 2), max(min(y1, H) - max(y0, 0), 2))
    try:
        cv2.grabCut(sub, gm, rect, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    a = np.where((gm == 1) | (gm == 3), 255, 0).astype(np.uint8)
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(a)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[k, cv2.CC_STAT_AREA] < 60:
        return None
    a = fill_holes(np.where(lab == k, 255, 0).astype(np.uint8))
    full = np.zeros((H, W), np.uint8); full[Y0:Y1, X0:X1] = a
    return full


def colour_mask(img, box, ring=10):
    """fallback when GrabCut fails on a low-contrast object (green chili on the teal table): Lab distance to the local
    table colour (median of a ring around the box), Otsu inside the box, largest component, holes filled"""
    H, W = img.shape[:2]; x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    X0, Y0, X1, Y1 = max(x0 - ring, 0), max(y0 - ring, 0), min(x1 + ring, W), min(y1 + ring, H)
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    sub = lab[Y0:Y1, X0:X1]; inner = np.zeros(sub.shape[:2], bool); inner[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = True
    if inner.all() or not inner.any():
        return None
    d8 = np.clip(np.linalg.norm(sub - np.median(sub[~inner], axis=0), axis=2) * 4, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(d8[inner], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ((d8 > max(t, 40)) & inner).astype(np.uint8) * 255
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab_, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    full = np.zeros((H, W), np.uint8); full[Y0:Y1, X0:X1] = fill_holes(np.where(lab_ == k, 255, 0).astype(np.uint8))
    return full


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def main():
    date, t0, t1 = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(OUT, exist_ok=True)
    data = fetch(date, t0, t1)
    print(len(data), 'sessions')
    jf = os.path.join(OUT, 'refs.json')
    refs = {r['id']: r for r in (json.load(open(jf)) if os.path.exists(jf) else [])}
    sheet = []
    for stem in sorted(data):
        recs = [json.loads(l) for l in data[stem]['jsonl'].decode().splitlines() if l.strip()]
        st = next((r for r in recs if r.get('event') == 'start'), None)
        det = next((r for r in recs if r.get('event') == 'detect' and r.get('why') == 'start'), None)
        if not st or not det or not det.get('target_box') or not det.get('dest_box'):
            print('skip (no cue)', stem); continue
        img = cv2.imdecode(np.frombuffer(data[stem]['cue'], np.uint8), cv2.IMREAD_COLOR)
        band = box_band(img.shape, det['target_box']) | box_band(img.shape, det['dest_box'])
        clean = cv2.inpaint(img, band, 5, cv2.INPAINT_TELEA)
        dets = sorted([d for d in det.get('dets', []) if d[5] >= MIN_CONF], key=lambda d: -d[5]); keep = []
        for d in dets:
            if all(iou(d[1:5], k[1:5]) < 0.6 for k in keep):
                keep.append(d)
        H, W = img.shape[:2]; alpha = np.zeros((H, W), np.uint8); edge = np.zeros((H, W, 4), np.uint8); objs = []
        for d in keep:
            bw, bh = d[3] - d[1], d[4] - d[2]
            o = {'name': d[0], 'conf': round(d[5], 3), 'box': [round(v, 1) for v in d[1:5]],
                 'cx': round((d[1] + d[3]) / 2, 1), 'cy': round((d[2] + d[4]) / 2, 1), 'ghost': False}
            mk = None if d[0] in NO_MASK else object_mask(clean, d[1:5])
            if mk is not None and (mk > 0).sum() < 0.12 * bw * bh:
                mk = None                                   # GrabCut failed (low contrast)
            if mk is None and d[0] not in NO_MASK:
                mk = colour_mask(clean, d[1:5])             # fallback: colour difference to the local table colour
                if mk is not None and not 0.08 * bw * bh <= (mk > 0).sum() <= 0.95 * bw * bh:
                    mk = None
                if mk is not None:
                    o['ghost_method'] = 'colour'
            if mk is None:
                objs.append(o); continue
            alpha |= mk
            cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(edge, cs, -1, (0, 0, 0, 200), 4); cv2.drawContours(edge, cs, -1, (0, 230, 255, 255), 2)  # BGRA: yellow on a dark rim
            ys, xs = np.where(mk > 0)
            o.update(cx=round(float(xs.mean()), 1), cy=round(float(ys.mean()), 1), area=int(len(xs)), ghost=True); objs.append(o)
        soft = cv2.GaussianBlur(alpha, (0, 0), 1.0)
        obj = np.dstack([clean * (soft[..., None] > 0), soft])          # RGB zeroed where transparent (small PNG)
        rid = stem
        cv2.imwrite(os.path.join(OUT, f'ref_{rid}.jpg'), clean, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(os.path.join(OUT, f'obj_{rid}.png'), obj)
        cv2.imwrite(os.path.join(OUT, f'edge_{rid}.png'), edge)
        g = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY), (0, 0), 1.4), 30, 90)
        g = cv2.dilate(g, np.ones((2, 2), np.uint8))
        line = np.zeros((H, W, 4), np.uint8); line[g > 0] = (255, 80, 255, 255)        # BGRA magenta
        cv2.imwrite(os.path.join(OUT, f'line_{rid}.png'), line)
        tb = det['target_box']; covered = min(tb[2] - tb[0], tb[3] - tb[1]) < 2 * LINE + 8
        p = st.get('parsed') or [None, None, None]
        end = next((r for r in recs if r.get('event') == 'end'), {})
        refs[rid] = {'id': rid, 'time': f'{stem[9:11]}:{stem[11:13]}:{stem[13:15]}', 'date': stem[:8], 'session': st.get('session_id'),
                     'task': st.get('task'), 'task_no': p[0], 'target': p[1], 'dest': p[2], 'reason': end.get('reason'),
                     'placed': end.get('placed'), 'objects': objs,
                     'target_box_center': [round((tb[0] + tb[2]) / 2, 1), round((tb[1] + tb[3]) / 2, 1)],
                     'dest_box_center': [round((det['dest_box'][0] + det['dest_box'][2]) / 2, 1), round((det['dest_box'][1] + det['dest_box'][3]) / 2, 1)],
                     'target_covered': bool(covered)}
        print(rid, st.get('task'), '|', ', '.join(o['name'] + ('' if o['ghost'] else '(label only)') for o in objs), '| TARGET COVERED' if covered else '')
        bg = np.full((H, W, 3), 60, np.uint8); bg[(np.indices((H, W)).sum(0) // 16) % 2 == 0] = 90
        a = soft[..., None] / 255.0; ghost = (clean * a + bg * (1 - a)).astype(np.uint8)
        e = edge[..., 3:] / 255.0; ghost = (ghost * (1 - e) + edge[..., :3] * e).astype(np.uint8)
        la = line[..., 3:] / 255.0; lines = (clean * 0.45 * (1 - la) + line[..., :3] * la).astype(np.uint8)
        row = np.hstack([img, clean, ghost, lines])
        cv2.putText(row, f"{refs[rid]['time']} {st.get('task')}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(row, f"{refs[rid]['time']} {st.get('task')}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        sheet.append(cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2), interpolation=cv2.INTER_AREA))
    out = sorted(refs.values(), key=lambda r: (r['date'], r['time']))
    with open(jf + '.part', 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(jf + '.part', jf)
    for i in range(0, len(sheet), 7):
        cv2.imwrite(os.path.join(OUT, f'check_{date}_{t0}_{i // 7 + 1}.jpg'), np.vstack(sheet[i:i + 7]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(len(out), 'references in', jf)


if __name__ == '__main__':
    main()
