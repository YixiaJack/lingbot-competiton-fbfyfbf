"""Trigger rate of the yolocue rule on recorded first frames (no robot needed): for every team episode, would the
online cue have been drawn on its frame 0, and is the white box on the object that actually moved?

Input: DIR/index.json + DIR/frames/<uid>_first.jpg, _last.jpg (experiments/autolabel/firstframes/ff_extract.py and
ff_seen.py on the dev machine). Runs on the 5080 with the inference env's python (yolocue_bridge next to this file).
- instruction -> target/destination with yolocue_bridge.parse; a collected instruction that names the object loosely
  ("apple") falls back to the label's custom field ("red apple");
- truth = real_check/diffgt.py on first/last frame (blob that turned into bare table), episodes marked 成功/部分成功;
- split by whether the episode was in the snapshot the YOLO labels came from (yolo_snapshot_ep) or recorded later.
usage: python -I ff_trigger.py DIR [--json OUT]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yolocue_bridge as YC  # noqa: E402


def parse_row(r):
    """(task, target, dest), how) -- how: 'instruction', 'custom field' or the reason it failed."""
    s = ' '.join(str(r.get('instruction') or '').split())
    pr = YC.parse(s)
    if pr:
        return pr, 'instruction'
    for task, pats, table, dests, prefix in YC._PATS:
        for p in pats:
            m = p.fullmatch(s)
            if not m or m.group(2).lower() not in dests:
                continue
            for v in (r.get('custom') or {}).values():
                if str(v).lower().strip() in table:
                    return (task, table[str(v).lower().strip()], prefix + m.group(2).lower()), 'custom field'
            return None, f'object not in the class list: {m.group(1).lower()}'
    return None, 'not a task-1/2 instruction'


def diffgt(first, last):
    """real_check/diffgt.py for one episode -> start box or None."""
    load = lambda p: np.asarray(Image.open(p).convert('RGB').filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
    a, b = load(first), load(last); H = a.shape[0]
    m = np.abs(a - b).max(2) > 45; m[:int(H * 0.25)] = False
    k = np.ones((3, 3), np.uint8); m8 = m.astype(np.uint8)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k, iterations=2); m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k, iterations=4)
    cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); m8 = np.zeros_like(m8); cv2.drawContours(m8, cs, -1, 1, -1)
    m = m8.astype(bool); n, lab, stats, _ = cv2.connectedComponentsWithStats(m8)
    low = np.zeros_like(m); low[H // 2:] = True
    if not (low & ~m).any():
        return None
    tab = np.median(a[low & ~m], axis=0)
    teal = lambda img, sl, comp: float((np.linalg.norm(img[sl][comp] - tab, axis=1) < 35).mean())
    comps = []
    for i in range(1, n):
        x0, y0, w, h, area = [int(v) for v in stats[i]]
        if area < 250:
            continue
        sl = (slice(y0, y0 + h), slice(x0, x0 + w)); comp = lab[sl] == i
        comps.append((teal(b, sl, comp) - teal(a, sl, comp), area, [x0, y0, x0 + w, y0 + h]))
    if not comps:
        return None
    gain, area, box = max(comps, key=lambda c: c[0] * np.sqrt(c[1]))
    return box if gain > 0.25 and area <= 20000 else None


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


ctr = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
inside = lambda p, b: b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]
match = lambda b, g: iou(b, g) >= 0.3 or (inside(ctr(b), g) and inside(ctr(g), b))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('dir', type=Path); ap.add_argument('--json', type=Path)
    a = ap.parse_args()
    rows = json.loads((a.dir / 'index.json').read_text())
    det = YC.detector(YC.load_config())
    res = []
    for r in rows:
        pr, how = parse_row(r)
        rec = {'uid': r['uid'], 'instruction': r.get('instruction'), 'result': r.get('result'), 'score': r.get('score'),
               'seen_by_yolo': r.get('yolo_snapshot_ep') is not None, 'parsed': pr, 'how': how}
        if pr:
            rgb = cv2.cvtColor(cv2.imread(str(a.dir / 'frames' / f"{r['uid']}_first.jpg")), cv2.COLOR_BGR2RGB)
            dets = det(rgb)
            tb, db, reason = YC.select(dets, *pr)
            tc = sorted((d[5] for d in dets if d[0] == pr[1]), reverse=True)
            dc = sorted((d[5] for d in dets if d[0] == pr[2]), reverse=True)
            rec.update(reason=reason, target_box=tb, dest_box=db, target_conf=tc[:2], dest_conf=dc[:1],
                       other=[[d[0], d[5]] for d in dets if d[5] >= 0.5 and d[0] not in pr[1:] and not d[0].startswith(('plate_', 'organizer_'))][:8])
            if r.get('result') in ('成功', '部分成功'):
                g = diffgt(a.dir / 'frames' / f"{r['uid']}_first.jpg", a.dir / 'frames' / f"{r['uid']}_last.jpg")
                rec['truth'] = g
                if g and tb:
                    rec['cue_right'] = bool(match(tb, g))
        res.append(rec)

    def table(sel):
        P = [x for x in res if x['parsed'] and sel(x)]
        ok = [x for x in P if x['reason'] == 'ok']
        chk = [x for x in ok if 'cue_right' in x]
        return {'episodes': len(P), 'cue': len(ok), 'rate': round(len(ok) / max(len(P), 1), 3),
                'reasons': dict(collections.Counter(x['reason'] for x in P)),
                'cue_with_truth': len(chk), 'cue_right': sum(x['cue_right'] for x in chk)}
    out = {'rows': len(res), 'parsed': sum(1 for x in res if x['parsed']),
           'unparsed': dict(collections.Counter(x['how'] for x in res if not x['parsed']).most_common(12)),
           'all': table(lambda x: True),
           'new (YOLO never saw)': table(lambda x: not x['seen_by_yolo']),
           'in YOLO snapshot': table(lambda x: x['seen_by_yolo']),
           'new task 1': table(lambda x: not x['seen_by_yolo'] and x['parsed'][0] == 1),
           'new task 2': table(lambda x: not x['seen_by_yolo'] and x['parsed'][0] == 2)}
    by = collections.defaultdict(lambda: [0, 0])
    for x in res:
        if x['parsed'] and not x['seen_by_yolo']:
            by[x['parsed'][1]][0] += 1; by[x['parsed'][1]][1] += x['reason'] == 'ok'
    out['new by target (n, cue)'] = {k: v for k, v in sorted(by.items(), key=lambda kv: (kv[1][1] / kv[1][0], -kv[1][0]))}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.json:
        a.json.write_text(json.dumps({'summary': out, 'episodes': res}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
