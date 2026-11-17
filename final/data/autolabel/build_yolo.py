"""Build a YOLO dataset from the LocateAnything labels (camera_top, official 647 real demos).

  build_yolo.py --out ~/locany/yolo/ds_v1 --levels 0

Split: the 45 `*_reserve*` episodes are val (never trained on), the rest train.
Per-frame labels:
  Task 1/2 (only the instructed target moves):
    - target: the frame's single-instance box, class from the instruction. The episode's start box (frame 0) must agree
      with the independent frame-0 grounding (ground0.json, IoU >= 0.5), else the episode is skipped. A frame is used
      only if its target box sits at the start box (not picked up yet) or at the end box (last frame), IoU >= 0.5:
      mid-air frames are dropped because the model then boxes a look-alike.
    - destination: frame-0 single-instance box (ground0.json) if the target's end box centre falls inside it; else the
      frame-0 plate box (task 1) / instructed level box (task 2) that holds it; else the episode is skipped.
    - task 1 other plates: frame-0 all-category plate boxes, colour from the pixels (nearest colour centroid of the
      validated destination plates; the model's own plate colours are unreliable on this camera).
    - task 2 (--t2 v2): organizer = frame-0 whole-organizer box split at 0.38 / 0.69 of its height (LocateAnything
      cannot tell the shelves apart); other cosmetics identified by appearance (cosm_id.json); episodes whose target
      box looks like another cosmetic are skipped.
    - other objects: frame-0 all-category boxes of classes whose predicted label was reliable in the diffgt check,
      dropped in frames where the box region changed (arm passing over it).
  Task 3: the frame's all-category boxes; frames with a box > 25% of the image (arms swallowed into a box) or duplicated
    brush/dustpan/basket are dropped.
"""
import argparse
import colorsys
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

H = Path.home() / 'locany'
DS = H / 'data' / os.environ.get('LA_DS', 'public647')
L = H / 'label'

T1_OBJ = ['avocado', 'banana', 'bell_pepper', 'carrot', 'cucumber', 'dragon_fruit', 'eggplant', 'green_apple', 'kiwi',
          'lemon', 'mango', 'mangosteen', 'mushroom', 'onion', 'orange', 'papaya', 'passion_fruit', 'peach', 'pear',
          'persimmon', 'pineapple', 'pomegranate', 'potato', 'pumpkin', 'red_apple', 'strawberry', 'tomato',
          'watermelon', 'wax_apple', 'grapes', 'cherry', 'tangerine', 'red_chili', 'red_delicious', 'starfruit', 'green_chili']
COLORS = ['white', 'yellow', 'pink', 'purple', 'blue']
COSM = ['lip_gloss', 'lotion_bottle', 'lipstick', 'contour', 'brush_large', 'highlighter', 'powder_puff', 'eyeshadow_palette',
        'brush_small', 'glitter_compact']   # last two only in the team datasets (11/14)
LEVELS = ['top', 'middle', 'bottom']
T3_OBJ = ['brush', 'dustpan', 'basket', 'paper_ball_red', 'paper_ball_yellow']
CLASSES = T1_OBJ + [f'plate_{c}' for c in COLORS] + COSM + [f'organizer_{l}' for l in LEVELS]   # 11/14: no task 3 in YOLO
CID = {c: i for i, c in enumerate(CLASSES)}

T1_PHRASE = {c.replace('_', ' '): c for c in T1_OBJ}
T1_PHRASE.update({'bunch of grapes': 'grapes', 'cherries': 'cherry', 'mandarin': 'tangerine',
                  'red chili pepper': 'red_chili', 'red delicious apple': 'red_delicious', 'kiwi fruit': 'kiwi',
                  'green chili pepper': 'green_chili'})
T2_PHRASE = {'beige lip gloss tube': 'lip_gloss', 'beige lotion pump bottle': 'lotion_bottle',
             'black lipstick with a clear cap': 'lipstick', 'dark-colored round contour compact': 'contour',
             'large black and gold makeup brush': 'brush_large', 'light-colored round highlighter compact': 'highlighter',
             'pink makeup puff': 'powder_puff', 'rectangular multicolor eyeshadow palette': 'eyeshadow_palette',
             'small black and gold makeup brush': 'brush_small', 'black round glitter compact': 'glitter_compact'}
# predicted-class precision >= ~0.75 in the frame-0 diffgt check of the all-category prompt (docs/11)
RELIABLE = {'cucumber', 'green_apple', 'bell_pepper', 'lemon', 'mangosteen', 'mushroom', 'strawberry', 'kiwi', 'pumpkin',
            'cherry', 'red_chili', 'eggplant', 'onion', 'carrot', 'grapes', 'banana', 'watermelon', 'potato', 'pineapple',
            'brush_large', 'lotion_bottle', 'eyeshadow_palette'}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def ctr(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def inside(p, b, pad=10):
    return b[0] - pad <= p[0] <= b[2] + pad and b[1] - pad <= p[1] <= b[3] + pad


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def img_path(e, j):
    p = L / 'main/images' / f'ep{e:03d}' / f'{j:04d}.jpg'
    return p if p.exists() else L / 'first0/images' / f'ep{e:03d}' / f'{j:04d}.jpg'


def crop_gray(path, b):
    im = Image.open(path).convert('L')
    return np.asarray(im.crop([int(v) for v in b]), dtype=np.float32)


def plate_feat(path, b):
    """median colour of the inner half of a plate box -> (s*cos h, s*sin h, v)"""
    w, h = b[2] - b[0], b[3] - b[1]
    inner = [b[0] + w / 4, b[1] + h / 4, b[2] - w / 4, b[3] - h / 4]
    a = np.asarray(Image.open(path).convert('RGB').crop([int(v) for v in inner]), dtype=np.float32).reshape(-1, 3) / 255
    r, g, bl = np.median(a, axis=0)
    hh, s, v = colorsys.rgb_to_hsv(r, g, bl)
    return np.array([s * np.cos(2 * np.pi * hh), s * np.sin(2 * np.pi * hh), v])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--levels', default='0')
    ap.add_argument('--t2', choices=['v1', 'v2'], default='v2',
                    help='v2: organizer split into thirds of the whole-organizer box + all 8 cosmetics from cosm0.json')
    ap.add_argument('--regions', action='store_true',
                    help='task 2 levels = where placed objects sit (fractions of the organizer box from 136 confirmed placements), '
                         'not full bands: shelf boards between levels are left out')
    ap.add_argument('--t3-levels', default=None, help='task-3 frames only from these labelling rounds (task 3 is easy and plentiful)')
    ap.add_argument('--t1id', action='store_true', help='task 1 other objects identified by appearance (fruit_id.json) first')
    ap.add_argument('--split', default='0.38,0.69', help='v2: organizer level boundaries as fractions of the box height; midpoints of the median placed-object heights per level (0.24, 0.53, 0.84) over 216 episodes')
    a = ap.parse_args()
    levels = {int(x) for x in a.levels.split(',')}

    meta = {}
    for r in jl(DS / 'meta/labels.jsonl'):
        lab = r['labels'][0]; cid = lab['custom_fields'][0]['value']; s = lab['task_instruction']; e = r['episode_index']
        # val: the official reserve episodes, plus every third team episode (held out to test on unseen scenes)
        t = int(cid[1]); m = {'task': t, 'id': cid, 'val': 'reserve' in cid or (e >= 1000 and (e - 1000) % 3 == 0)}
        mm = None
        if t == 1:
            mm = re.search(r'(?:Pick up|[Pp]ut|Place|Move|[Cc]ould you put) the (.+?)(?: and place it)? (?:on|onto) the (\w+) plate', s)
            if mm and mm.group(1) in T1_PHRASE: m.update(tgt=T1_PHRASE[mm.group(1)], dest=f'plate_{mm.group(2)}')
        elif t == 2:
            mm = re.search(r'the (.+?)(?: and put it)? (?:in|into|inside) the (top|middle|bottom)', s)
            if mm and mm.group(1) in T2_PHRASE: m.update(tgt=T2_PHRASE[mm.group(1)], dest=f'organizer_{mm.group(2)}')
        if t in (1, 2) and 'tgt' not in m:
            m['task'] = 0   # e.g. 'the apple' (which apple?) or no instruction: not used
        meta[e] = m

    g0 = {int(k): v for k, v in json.load(open(L / 'ground0.json')).items()}
    lv0 = {int(k): v for k, v in json.load(open(L / 'levels0.json')).items()} if (L / 'levels0.json').exists() else {}
    PH_ORG = 'clear acrylic organizer'
    C2 = {'beige lip gloss tube': 'lip_gloss', 'beige lotion pump bottle': 'lotion_bottle',
          'black lipstick with a clear cap': 'lipstick', 'dark-colored round contour compact': 'contour',
          'large black and gold makeup brush': 'brush_large', 'light-colored round highlighter compact': 'highlighter',
          'pink makeup puff': 'powder_puff', 'rectangular multicolor eyeshadow palette': 'eyeshadow_palette'}
    cosm0 = {int(k): v for k, v in json.load(open(L / 'cosm0.json')).items()} if a.t2 == 'v2' else {}
    cid_raw = json.load(open(L / 'cosm_id.json')) if a.t2 == 'v2' else {}
    bad_t2 = set(cid_raw.pop('_bad_target_eps', []))
    cosm_id = {int(k): v for k, v in cid_raw.items()}
    f1, f2 = (float(x) for x in a.split.split(','))

    REG = {'top': (0.12, 0.07, 0.88, 0.32), 'middle': (0.12, 0.36, 0.88, 0.64), 'bottom': (0.12, 0.71, 0.88, 0.96)}
    fid_raw = json.load(open(L / 'fruit_id.json')) if a.t1id else {}
    bad_t1 = set(fid_raw.pop('_bad_target_eps', []))
    fruit_id = {int(k): v for k, v in fid_raw.items()}

    def thirds(e):
        s_ = cosm0.get(e, {}).get(PH_ORG, [])
        if len(s_) < 2 or not all(s_) or iou(s_[0], s_[1]) < 0.7:
            return None
        x1, y1, x2, y2 = s_[0]; hh = y2 - y1
        if a.regions:
            ww = x2 - x1
            return {k: [x1 + r[0] * ww, y1 + r[1] * hh, x1 + r[2] * ww, y1 + r[3] * hh] for k, r in REG.items()}
        return {'top': [x1, y1, x2, y1 + f1 * hh], 'middle': [x1, y1 + f1 * hh, x2, y1 + f2 * hh], 'bottom': [x1, y1 + f2 * hh, x2, y2]}

    def cosmetics(e, start, org):
        acc = {}
        for ph, cls in C2.items():
            s_ = cosm0.get(e, {}).get(ph, [])
            if len(s_) < 2 or not all(s_) or iou(s_[0], s_[1]) < 0.5:
                stats['cosm_unstable'] += 1; continue
            b = s_[0]
            if (b[2] - b[0]) * (b[3] - b[1]) > 0.03 * 640 * 480 or (org and inside(ctr(b), org, 0)):
                stats['cosm_big_or_in_organizer'] += 1; continue
            acc[cls] = b
        tgt = meta[e]['tgt']
        acc.pop(tgt, None)   # the target comes from the confirmed start box
        acc['__target__'] = start
        bad = set()
        ks = list(acc)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                if iou(acc[ks[i]], acc[ks[j]]) >= 0.3:
                    bad |= {ks[i], ks[j]} - {'__target__'}
        stats['cosm_conflict'] += len(bad)
        out_ = [(c, b) for c, b in acc.items() if c != '__target__' and c not in bad]
        stats['cosm_kept'] += len(out_)
        return out_

    multi0 = {}
    for r in jl(L / 'first0/labels.jsonl'):
        if r['task'] in (1, 3):
            multi0[r['ep']] = r['boxes']
    for r in jl(L / 'first0_t2/labels.jsonl'):
        multi0[r['ep']] = r['boxes']
    rows = defaultdict(dict)
    for r in jl(L / 'main/labels.jsonl'):
        if r['level'] in levels or r['frame'] == 0:
            rows[r['ep']][r['frame']] = r

    stats = Counter()
    episode_info = {}
    # pass 1: task 1/2 episode-level boxes
    for e, m in meta.items():
        if m['task'] in (0, 3) or e not in rows:
            continue
        R = rows[e]; fr = sorted(R)
        if 0 not in R or not R[0].get('target_box') or not g0.get(e, {}).get('tgt'):
            stats['skip_no_start'] += 1; continue
        start = R[0]['target_box']
        if iou(start, g0[e]['tgt'][0]) < 0.5:
            stats['skip_start_disagree'] += 1; continue
        end = R[fr[-1]].get('target_box') if len(fr) > 2 else None
        if not end:
            stats['skip_no_end'] += 1; continue
        # the destination must contain where the target ends up: frame-0 grounding first, else the frame-0 box that
        # holds the placed target (task 1: a plate box of the all-category prompt; task 2: the instructed level box)
        dest = (g0[e].get('dest') or [None])[0]
        if m['task'] == 1 and e in bad_t1:
            stats['skip_t1_target_looks_wrong'] += 1; continue
        if m['task'] == 2 and a.t2 == 'v2':
            if e in bad_t2:
                stats['skip_t2_target_looks_wrong'] += 1; continue
            th3 = thirds(e)
            if not th3:
                stats['skip_no_organizer'] += 1; continue
            dest = th3[m['dest'].split('_')[1]]
            if not inside(ctr(end), dest, 10 if not a.regions else 15):
                stats['skip_dest_miss_t2'] += 1; continue
            episode_info[e] = dict(start=start, end=end, dest=dest, thirds=th3)
            stats['episodes_ok_t2'] += 1
            continue
        if not dest or not inside(ctr(end), dest):
            if m['task'] == 1:
                cand = [x[1:] for x in multi0.get(e, []) if x[0].startswith('plate_') and inside(ctr(end), x[1:], 0)]
            else:
                lvb = lv0.get(e, {}).get(m['dest'].split('_')[1])
                cand = [lvb] if lvb and inside(ctr(end), lvb, 0) else []
            if len(cand) != 1:
                stats['skip_dest_miss'] += 1; continue
            dest = cand[0]; stats['dest_from_placement'] += 1
        episode_info[e] = dict(start=start, end=end, dest=dest)
        stats['episodes_ok_t%d' % m['task']] += 1

    # plate colour centroids from the validated destination plates, per domain (team scenes are lit differently)
    dom = lambda e: 'team' if e >= 1000 else 'pub'
    feats = defaultdict(list)
    for e, inf in episode_info.items():
        if meta[e]['task'] == 1:
            feats[(dom(e), meta[e]['dest'])].append(plate_feat(img_path(e, 0), inf['dest']))
    cent = {}
    for d in ('pub', 'team'):
        for c in [f'plate_{x}' for x in COLORS]:
            v = feats.get((d, c), [])
            if len(v) >= 3: cent[(d, c)] = np.mean(v, axis=0)
            elif (('pub', c)) in cent: cent[(d, c)] = cent[('pub', c)]
    print('plate centroids from', {f'{k[0]}:{k[1]}': len(v) for k, v in feats.items()})

    def plate_color(e, b, used):
        """nearest colour centroid among the colours not already on this table (plates on a table differ in colour);
        never 'unsure': an unlabelled plate teaches the detector that plates are background (v4 on team scenes)"""
        f = plate_feat(img_path(e, 0), b)
        d = sorted((float(np.linalg.norm(f - c)), k[1]) for k, c in cent.items() if k[0] == dom(e) and k[1] not in used)
        return d[0][1] if d else None

    # static (frame-0) boxes per episode
    static = {}
    for e, inf in episode_info.items():
        t = meta[e]['task']; keep = [(meta[e]['dest'], inf['dest'], 'static')]
        if t == 2 and a.t2 == 'v2':
            th3 = inf['thirds']; org = [th3['top'][0], th3['top'][1], th3['bottom'][2], th3['bottom'][3]]
            keep = [(f'organizer_{l}', th3[l], 'static') for l in LEVELS]
            # other cosmetics identified by appearance (cosm_id.py: kNN against confirmed targets, LOO 98% when accepted)
            keep += [(c, b, 'check') for c, b, _ in cosm_id.get(e, [])]
            stats['cosm_kept'] += len(cosm_id.get(e, []))
            static[e] = keep
            continue
        if t == 2 and e in lv0:
            L3 = lv0[e]
            if all(L3.get(l) for l in LEVELS) and ctr(L3['top'])[1] < ctr(L3['middle'])[1] < ctr(L3['bottom'])[1]:
                keep = [(f'organizer_{l}', L3[l], 'static') for l in LEVELS]
                stats['t2_levels_ok'] += 1
            else:
                stats['t2_levels_bad'] += 1
        if t == 1 and a.t1id:
            for cls, b, _ in fruit_id.get(e, []):
                if iou(b, inf['start']) < 0.3 and not any(iou(b, k[1]) >= 0.3 for k in keep):
                    keep.append((cls, b, 'check')); stats['fruit_id_kept'] += 1
        for cls, b in [(x[0], x[1:]) for x in multi0.get(e, [])]:
            if iou(b, inf['start']) >= 0.3 or any(iou(b, k[1]) >= 0.3 for k in keep):
                continue
            if cls.startswith('plate_'):
                c = plate_color(e, b, {k[0] for k in keep if k[0].startswith('plate_')})
                if c:
                    keep.append((c, b, 'static')); stats['plate_recoloured_' + ('same' if c == cls else 'changed')] += 1
                else:
                    stats['plate_ambiguous'] += 1
            elif cls in RELIABLE and cls != meta[e]['tgt']:
                keep.append((cls, b, 'check')); stats['distractor_kept'] += 1
            else:
                stats['distractor_dropped'] += 1
        static[e] = keep

    out = a.out
    if out.exists():
        shutil.rmtree(out)
    for sp in ('train', 'val'):
        (out / 'images' / sp).mkdir(parents=True); (out / 'labels' / sp).mkdir(parents=True)
    n_written = Counter(); cls_count = Counter()

    def write(e, j, w, h, boxes):
        sp = 'val' if meta[e]['val'] else 'train'
        name = f'ep{e:03d}_{j:04d}'
        os.symlink(img_path(e, j), out / 'images' / sp / f'{name}.jpg')
        with open(out / 'labels' / sp / f'{name}.txt', 'w') as f:
            for cls, b in boxes:
                x1, y1, x2, y2 = max(0, b[0]), max(0, b[1]), min(w, b[2]), min(h, b[3])
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                f.write(f'{CID[cls]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n')
                cls_count[(sp, cls)] += 1
        n_written[(sp, meta[e]['task'])] += 1

    for e, m in meta.items():
        R = rows.get(e, {})
        if m['task'] in (0, 3):   # task 3 is not part of the detector (user, 11/14)
            continue
        if m['task'] == 3:
            t3lv = {int(x) for x in a.t3_levels.split(',')} if a.t3_levels else None
            for j, r in sorted(R.items()):
                if t3lv is not None and r['level'] not in t3lv:
                    continue
                if r['bad'] or not r['boxes']:
                    stats['t3_bad'] += 1; continue
                bx = [(x[0], x[1:]) for x in r['boxes']]
                if any(c not in CID for c, _ in bx):
                    stats['t3_unknown'] += 1; continue
                big = any((b[2] - b[0]) * (b[3] - b[1]) > 0.25 * r['w'] * r['h'] for _, b in bx)
                dup = any(n > 1 for c, n in Counter(c for c, _ in bx).items() if c in ('brush', 'dustpan', 'basket'))
                if big or dup:
                    stats['t3_dropped_' + ('big' if big else 'dup')] += 1; continue
                write(e, j, r['w'], r['h'], bx)
            continue
        if e not in episode_info:
            continue
        inf = episode_info[e]
        g_ref = {}
        for j, r in sorted(R.items()):
            tb = r.get('target_box')
            if r['bad'] or not tb or not (iou(tb, inf['start']) >= 0.5 or iou(tb, inf['end']) >= 0.5):
                stats['t12_frame_dropped'] += 1; continue
            boxes = [(m['tgt'], tb)]
            for cls, b, kind in static[e]:
                if kind == 'check' and j != 0:
                    ref = g_ref.setdefault(tuple(b), crop_gray(img_path(e, 0), b))
                    if np.abs(crop_gray(img_path(e, j), b) - ref).mean() > 12:
                        stats['distractor_occluded'] += 1; continue
                boxes.append((cls, b))
            write(e, j, r['w'], r['h'], boxes)

    names = {i: c for i, c in enumerate(CLASSES)}
    (out / 'dataset.yaml').write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n" + ''.join(f"  {i}: {c}\n" for i, c in names.items()))
    report = {'stats': dict(stats), 'images': {f'{k[0]}_t{k[1]}': v for k, v in sorted(n_written.items())},
              'boxes_train': {c: cls_count[('train', c)] for c in CLASSES},
              'boxes_val': {c: cls_count[('val', c)] for c in CLASSES}}
    (out / 'build_report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({'stats': report['stats'], 'images': report['images']}, indent=1))
    thin = [c for c in CLASSES if cls_count[('train', c)] < 30]
    print('classes with < 30 train boxes:', thin)


if __name__ == '__main__':
    main()
