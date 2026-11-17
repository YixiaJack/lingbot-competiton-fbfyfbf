"""The question that matters on site: given the evaluator's instruction and frame zero, does the detector + rule
pipeline hand the 6B model the right target and destination boxes?

On the val renders (disjoint seeds) of tasks 1 and 2, with every task object fully visible (as the rules promise):
correct = IoU >= 0.5 with the true target AND with a true destination (any plate of the named colour / the named level);
fallback = no cue given (policy gets the plain instruction); wrong = a cue given but pointing at the wrong thing (worst
case, it misleads the policy). Reported per task, per look-alike group, per target class, for three thresholds.
Usage: select_eval.py --root R
"""
import argparse, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO
from catalog import CLASSES, CONFUSE, DET_ITEMS
from cues_real import parse_instruction, select_boxes, run_detector


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def split_organizer(dets):
    """Open-vocabulary detectors see one organizer: cut its box into three horizontal bands (head camera: the stepped
    organizer's bottom level is nearest the robot, i.e. lowest in the image; the top level is highest)."""
    out = []
    for k, c, b in dets:
        if k != 'organizer':
            out.append((k, c, b)); continue
        x1, y1, x2, y2 = b; h = (y2 - y1) / 3
        out += [('organizer_top', c, [x1, y1, x2, y1 + h]), ('organizer_middle', c, [x1, y1 + h, x2, y1 + 2 * h]),
                ('organizer_bottom', c, [x1, y1 + 2 * h, x2, y2])]
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--weights', default=None, help='default: our trained yolo11s'); p.add_argument('--tag', default='')
    p.add_argument('--classes', default=None, help='classes.json mapping prompts to class keys (YOLO-World)')
    a = p.parse_args()
    D = a.root / 'det'
    model = YOLO(a.weights or str(D / 'runs' / 'yolo11s' / 'weights' / 'best.pt'))
    names = model.names
    if a.classes:  # YOLO-World names are the prompt strings
        cj = json.loads(Path(a.classes).read_text()); to_key = dict(zip(cj['prompts'], cj['keys']))
        names = {i: to_key.get(n, n) for i, n in names.items()}
    rows, skipped, times = [], defaultdict(int), []
    for mp in sorted((D / 'meta' / 'val').glob('*.json')):
        if int(mp.name[:4]) >= DET_ITEMS:   # not part of the design (items 90-109 only)
            continue
        m = json.loads(mp.read_text())
        if m['spec']['task'] not in (1, 2):
            continue
        parsed = parse_instruction(m['instruction'])
        gt_t = [b for b in m['boxes'] if b['cls'] == m['spec']['target']]
        dcls = f"plate_{m['spec']['dest']}" if m['spec']['task'] == 1 else f"organizer_{m['spec']['dest']}"
        gt_d = [b for b in m['boxes'] if b['cls'] == dcls]
        if not gt_t or not gt_d or any(b['truncated'] for b in m['boxes'] if b['cls'] not in ('wall',)):
            skipped['object cut or hidden (rules promise full visibility)'] += 1
            continue
        img = np.asarray(Image.open(D / 'images' / 'val' / (mp.stem + '.png')).convert('RGB'))
        s = time.perf_counter(); dets = split_organizer(run_detector(model, img, names)); times.append(1000 * (time.perf_counter() - s))
        for thr in (0.3, 0.5, 0.7):
            tb, db, reason = select_boxes(dets, parsed, conf_target=thr, conf_dest=thr)
            if tb is None:
                outcome = 'fallback'
            else:
                ok_t = iou(tb, gt_t[0]['box']) >= 0.5
                ok_d = any(iou(db, g['box']) >= 0.5 for g in gt_d)
                outcome = 'correct' if ok_t and ok_d else 'wrong_target' if not ok_t else 'wrong_destination'
            rows.append({'img': mp.stem, 'task': m['spec']['task'], 'target': m['spec']['target'], 'dest': dcls,
                         'thr': thr, 'outcome': outcome, 'reason': reason, 'parsed': parsed is not None})
    def summary(sel):
        n = len(sel); c = defaultdict(int)
        for r in sel: c[r['outcome']] += 1
        return {'n': n, **{k: c[k] for k in ('correct', 'fallback', 'wrong_target', 'wrong_destination')},
                'correct_rate': c['correct'] / n if n else None,
                'wrong_rate': (c['wrong_target'] + c['wrong_destination']) / n if n else None}
    out = {'parse_failures': sum(not r['parsed'] for r in rows), 'skipped': dict(skipped),
           'cpu_ms_median': float(np.median(times)) if times else None, 'by_threshold': {}}
    for thr in (0.3, 0.5, 0.7):
        R = [r for r in rows if r['thr'] == thr]
        out['by_threshold'][str(thr)] = {
            'all': summary(R), 'task1': summary([r for r in R if r['task'] == 1]), 'task2': summary([r for r in R if r['task'] == 2]),
            'by_group': {'/'.join(sorted(g)): summary([r for r in R if r['target'] in g]) for g in CONFUSE},
            'by_target': {k: summary([r for r in R if r['target'] == k]) for k in sorted({r['target'] for r in R})},
            'fallback_reasons': {k: sum(1 for r in R if r['reason'] == k) for k in sorted({r['reason'] for r in R})}}
    out['detector'] = a.weights or 'our yolo11s (59 classes, trained on the renders)'
    (D / f'selection{a.tag}.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({t: v['all'] for t, v in out['by_threshold'].items()}, indent=1))


if __name__ == '__main__':
    main()
