"""Data for the board's field page (lingbot-board /field): one record per episode of one or more LeRobot v3 datasets.

Per episode: the head camera's first frame (thumbnail), detector boxes on it, the instruction's task / target /
destination, where the detector found them, the episode length and how far the robot state ends from where it
started (a robot that did not go home ends far away). Per task: the median of the first frames, i.e. the empty
table, as the background of the coverage map. With --gt (our simulated demos) the target detection is checked
against the renderer's boxes.

  python field_build.py --dataset /path/lerobot_ds[:source[:collector]] [--dataset ...] --weights best.pt --out OUT
         [--camera observation.images.cam_high] [--gt EPISODES_DIR] [--conf 0.25] [--title TEXT]

source: demo | dagger | rollout (default demo; rollout = the model's own runs: self-tests, inference);
collector: free text (who collected it). Output: OUT/field_data.json,
OUT/thumbs/<n>.jpg, OUT/bg_t<k>.jpg. Needs numpy, pyarrow, av, pillow, ultralytics.
With --seen FILE each episode gets the time it was first seen (the field page groups episodes into shifts by it)
unless the dataset carries its own collection time; dur_s = episode length / fps (the recording time).
"""
import argparse
import datetime as dt
import glob
import json
import re
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

PLATE_COLORS = ['white', 'yellow', 'pink', 'purple', 'blue']
LEVELS = ['top', 'middle', 'bottom']
T1_WORDS = ('plate',)
T2_WORDS = ('compartment', 'organizer', 'level', 'section')
T3_WORDS = ('brush and dustpan', 'dustpan', 'sweep', 'paper ball')
# instruction words that differ from the detector's class names
ALIASES = {'grape': 'grapes', 'chili pepper': 'chili', 'lotion pump bottle': 'lotion_bottle', 'lotion bottle': 'lotion_bottle',
           'lip gloss tube': 'lip_gloss', 'makeup brush': 'brush', 'makeup puff': 'powder_puff', 'powder puff': 'powder_puff',
           'round highlighter compact': 'highlighter', 'highlighter compact': 'highlighter', 'contour compact': 'contour',
           'glitter compact': 'glitter_compact', 'eyeshadow palette': 'eyeshadow_palette', 'red delicious': 'red_delicious',
           'wax apple': 'wax_apple', 'passion fruit': 'passion_fruit', 'dragon fruit': 'dragon_fruit', 'bell pepper': 'bell_pepper',
           'mandarin': 'tangerine', 'cherries': 'cherry'}   # 11/14: words used in the official 647 demos


def task_of(instr):
    s = instr.lower()
    if any(w in s for w in T3_WORDS):
        return 3
    if any(w in s for w in T2_WORDS):
        return 2
    if any(w in s for w in T1_WORDS):
        return 1
    return 0


def object_phrase(instr, task):
    s = instr.lower().rstrip('.?! ')
    if task == 1:
        m = re.search(r'(?:put|pick up|move|place)\s+the\s+(.+?)\s+(?:and place it\s+)?(?:on|onto)\s+the\s+(\w+)\s+plate', s)
        return (m.group(1), m.group(2)) if m else (None, next((c for c in PLATE_COLORS if c in s), None))
    if task == 2:
        m = re.search(r'(?:put|place|store|take|organize)\s+the\s+(.+?)\s+(?:and put it\s+)?(?:in|inside|into)\s+the\s+(\w+)', s)
        return (m.group(1), m.group(2)) if m else (None, next((lv for lv in LEVELS if lv in s), None))
    return None, None


def match_class(phrase, names, task):
    """Detector class for an instruction's object phrase: alias first, then the class whose words overlap most."""
    if not phrase:
        return None
    cand = [n for n in names if (task == 1 and not n.startswith(('plate_', 'organizer_')) and n not in COSMETIC)
            or (task == 2 and n in COSMETIC)]
    p = phrase.replace('-', ' ')
    for k, v in sorted(ALIASES.items(), key=lambda kv: -len(kv[0])):
        if k in p:
            p = p.replace(k, v.replace('_', ' '))
    if task == 2 and 'brush' in p:
        return 'brush_large' if 'large' in p else 'brush_small' if 'small' in p else 'brush_large'
    words = set(p.split())
    best, score = None, 0
    for n in cand:
        nw = set(n.split('_'))
        sc = len(words & nw) / len(nw) + (0.5 if n.replace('_', ' ') in p else 0) \
            + (1.0 if n.replace('_', '') in p.replace(' ', '') else 0)   # 'star fruit' -> starfruit
        if sc > score:
            best, score = n, sc
    return best if score >= 0.5 else None


# per-episode columns some platforms add with the collection time (checked in this order; else: first time we saw it)
TIME_COLUMNS = ('collect_time', 'record_time', 'start_time', 'created_at', 'create_time', 'timestamp_start')
COSMETIC = {'lotion_bottle', 'lip_gloss', 'lipstick', 'eyeshadow_palette', 'brush_large', 'brush_small', 'highlighter',
            'contour', 'powder_puff', 'glitter_compact'}


def frames_at(ds, cam, rows, where='first'):
    """{episode_index: HxWx3 uint8}: the camera's first (or last) frame of each episode (LeRobot v3 concatenated videos)."""
    info = json.loads((ds / 'meta/info.json').read_text())
    tmpl, fps = info['video_path'], info['fps']
    out, by_file = {}, {}
    for r in rows:
        key = (r[f'videos/{cam}/chunk_index'], r[f'videos/{cam}/file_index'])
        ts = r[f'videos/{cam}/from_timestamp'] if where == 'first' else r[f'videos/{cam}/to_timestamp'] - 1.0 / fps
        by_file.setdefault(key, []).append((ts, r['episode_index']))
    for (ci, fi), eps in by_file.items():
        path = ds / tmpl.format(video_key=cam, chunk_index=ci, file_index=fi)
        with av.open(str(path)) as c:
            st = c.streams.video[0]
            for ts, ep in sorted(eps):
                c.seek(int(max(ts - 1.0, 0) / st.time_base), stream=st, backward=True)
                last = None
                for fr in c.decode(st):
                    if fr.time is None: continue
                    if fr.time >= ts - 0.5 / fps:
                        last = fr; break
                    last = fr          # the last episode of a file may end before ts: keep the latest frame seen
                if last is not None:
                    out[ep] = last.to_ndarray(format='rgb24')
    return out


def first_frames(ds, cam, rows):
    return frames_at(ds, cam, rows, 'first')


def states(ds, rows, key='observation.state'):
    """{episode_index: T x D float array} of the robot state from the data parquet files."""
    tmpl = json.loads((ds / 'meta/info.json').read_text())['data_path']
    cache, out = {}, {}
    for r in rows:
        f = ds / tmpl.format(chunk_index=r['data/chunk_index'], file_index=r['data/file_index'])
        if f not in cache:
            t = pq.read_table(f, columns=['index', key])
            cache[f] = (np.asarray(t.column('index')), np.stack(t.column(key).to_numpy(zero_copy_only=False)))
        idx, st = cache[f]
        a = np.searchsorted(idx, r['dataset_from_index']); b = np.searchsorted(idx, r['dataset_to_index'])
        if b > a:
            out[r['episode_index']] = st[a:b].astype(np.float64)
    return out


ARMS, GRIPS = (list(range(0, 6)), list(range(7, 13))), (6, 13)   # ALOHA / Cobot Magic 14-d: 6 joints + gripper per arm


def gripper_events(x):
    """(close frame indices, open frame indices) of one gripper: closed = more than half the episode's range away from
    where it started (the sign and units of the gripper differ between robots, so nothing is hard-coded)."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-3:
        return [], []
    closed = np.abs(x - x[0]) > 0.5 * (hi - lo)
    e = np.diff(closed.astype(int))
    return list(np.where(e == 1)[0] + 1), list(np.where(e == -1)[0] + 1)


def traj_metrics(q, fps, task):
    """Phases of one episode from the joint states (seconds): wait before moving, reach until the gripper closes,
    carry until it opens, return until the arm is back near its start; regrasps, pauses, which arm."""
    T, D = q.shape
    if T < 5 or D < 14:
        return {}
    act = int(np.argmax([np.abs(np.diff(q[:, a], axis=0)).sum() for a in ARMS]))
    sp = np.linalg.norm(np.diff(q[:, ARMS[act]], axis=0), axis=1) * fps
    thr = max(0.02, 0.1 * float(np.percentile(sp, 95)))
    moving = sp > thr
    t_move = int(np.argmax(moving)) if moving.any() else None
    d0 = np.linalg.norm(q[:, ARMS[act]] - q[0, ARMS[act]], axis=1)
    m = {'arm': 'left' if act == 0 else 'right', 'total': round(T / fps, 2),
         'wait': round(t_move / fps, 2) if t_move is not None else None}
    if task in (1, 2):
        c, o = gripper_events(q[:, GRIPS[act]])
        grasp = c[0] if c else None
        rel = next((x for x in reversed(o) if grasp is not None and x > grasp), None)
        home = None
        if rel is not None:
            near = np.where(d0[rel:] < 0.15 * max(float(d0.max()), 1e-6))[0]
            home = rel + int(near[0]) if len(near) else None
        m.update({'closes': len(c), 'regrasp': max(0, len(c) - 1),
                  'reach': round((grasp - (t_move or 0)) / fps, 2) if grasp is not None else None,
                  'carry': round((rel - grasp) / fps, 2) if rel is not None else None,
                  'back': round((home - rel) / fps, 2) if home is not None else None})
        end = home if home is not None else T - 1
    else:
        cl = [gripper_events(q[:, g])[0] for g in GRIPS]
        both = max(x[0] for x in cl) if all(cl) else None
        m.update({'closes': [len(x) for x in cl], 'grasp_tools': round(both / fps, 2) if both is not None else None})
        end = T - 1
    start = t_move or 0
    if end > start:
        still = ~moving[start:end]
        runs, cur = [], 0
        for v in still:
            cur = cur + 1 if v else 0
            runs.append(cur)
        m['pause'] = round(float(still.sum()) / fps, 2)
        m['longest_pause'] = round(max(runs) / fps, 2) if runs else 0.0
    m['home_dist'] = round(float(d0[-1] / max(float(d0.max()), 1e-6)), 3)   # 0 = ended where it started
    return m


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def inside(pt, box, pad=0.1):
    w, h = box[2] - box[0], box[3] - box[1]
    return box[0] - pad * w <= pt[0] <= box[2] + pad * w and box[1] - pad * h <= pt[1] <= box[3] + pad * h


def outcome(task, target, dest, d0, d1, width):
    """Where things ended up, from the first and the last frame's detections.
    Identity comes from frame 0 (nothing in the way); the last frame only says what left its place and where something
    new appeared, because at the end an object can sit under the gripper or be confused with a look-alike.
    Tasks 1, 2: ok / wrong_plate / wrong_level / not_placed / wrong_object / not_moved / target_not_seen_end.
    Task 3: ok / partial / not_swept."""
    thr = 0.04 * width
    def best(ds, cls):
        c = [d for d in ds if d[0] == cls]
        return max(c, key=lambda d: d[5]) if c else None
    if task == 3:
        basket = best(d1, 'basket') or best(d0, 'basket')
        b0 = [d for d in d0 if d[0].startswith('paper_ball')]
        b1 = [d for d in d1 if d[0].startswith('paper_ball') and not (basket and inside(center(d[1:5]), basket[1:5], 0.05))]
        if not b0: return 'unknown', []
        return ('ok' if not b1 else 'partial' if len(b1) < len(b0) else 'not_swept'), []
    if not target:
        return 'unknown', []
    fixed = ('plate_', 'organizer_')
    mov0 = [d for d in d0 if not d[0].startswith(fixed)]
    if target not in [d[0] for d in mov0]:
        return 'unknown', []          # the target was not found at frame 0: who moved cannot be told
    mov1 = [d for d in d1 if not d[0].startswith(fixed)]
    near = lambda a, b: np.hypot(*np.subtract(center(a[1:5]), center(b[1:5]))) <= thr
    stayed1 = set()
    left = []                                   # frame-0 objects no longer at their place
    for d in mov0:
        # still there: the same label nearby, or another label on almost the same box (look-alikes get relabelled;
        # a gripper hovering over the old spot does not count, its box differs)
        k = next((i for i, e in enumerate(mov1) if i not in stayed1 and near(d, e) and (e[0] == d[0] or iou(d[1:5], e[1:5]) >= 0.5)), None)
        if k is None: left.append(d[0])
        else: stayed1.add(k)
    arrived = [e for i, e in enumerate(mov1) if i not in stayed1]   # something where nothing was at frame 0
    pref = 'plate_' if task == 1 else 'organizer_'
    holders = [d for d in d0 if d[0].startswith(pref)]   # plates and the organizer do not move: frame 0, where nothing covers them
    def holder_of(e):
        h = [d for d in holders if inside(center(e[1:5]), d[1:5], 0.05 if task == 2 else 0.1)]
        return h[0][0] if h else None
    if target in left:
        if not arrived:
            return 'target_not_seen_end', left
        dest_cls = f'{pref}{dest}'
        same = [e for e in arrived if e[0] == target]      # the target seen at its new place under its own label
        if any(holder_of(e) == dest_cls for e in (same or arrived)):
            return 'ok', left
        if same:   # only call it wrong when the target itself is seen elsewhere: a false alarm would get a good demo deleted
            return (('wrong_plate' if task == 1 else 'wrong_level') if any(holder_of(e) for e in same) else 'not_placed'), left
        return 'target_not_seen_end', left
    if left:
        return 'wrong_object', left
    return 'not_moved', left


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', action='append', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--camera')
    p.add_argument('--gt', type=Path, help='our simulated episodes dir (complete.json per episode), sim check only')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--title', default='')
    p.add_argument('--seen', type=Path, help='JSON file remembering when each episode was first seen (kept across runs)')
    a = p.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.weights)
    names = [model.names[i] for i in range(len(model.names))]
    (a.out / 'thumbs').mkdir(parents=True, exist_ok=True)
    episodes, firsts = [], {1: [], 2: [], 3: []}
    seen = json.loads(a.seen.read_text()) if a.seen and a.seen.exists() else {}
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    gt_stats = {'checked': 0, 'target_hit': 0, 'target_missed': 0, 'target_wrong_place': 0}
    n = 0
    for spec in a.dataset:
        path, source, collector = (spec.split(':') + ['demo', ''])[:3]
        ds = Path(path)
        info = json.loads((ds / 'meta/info.json').read_text()); fps = info.get('fps') or 30
        cams = [k for k, v in info['features'].items() if v.get('dtype') == 'video']
        cam = a.camera or next((k for k in cams if any(w in k for w in ('high', 'top', 'head'))), cams[0])
        rows = []
        for f in sorted(glob.glob(str(ds / 'meta/episodes/*/*.parquet'))):
            rows += pq.read_table(f).to_pylist()
        frames = first_frames(ds, cam, rows)
        lasts = frames_at(ds, cam, rows, 'last')
        sts = states(ds, rows)
        eps = [r for r in rows if r['episode_index'] in frames]
        def detect(imgs):   # small batches, keep only the boxes (fits a 3 GB login node)
            out = []
            for i in range(0, len(imgs), 16):
                # PIL images: ultralytics reads numpy arrays as BGR, our frames are RGB (swapped colours = missed fruits)
                for pr in model.predict([Image.fromarray(x) for x in imgs[i:i + 16]], conf=a.conf, verbose=False):
                    b = pr.boxes
                    out.append([[names[int(c)], *[round(float(v), 1) for v in xyxy], round(float(s), 3)]
                                for xyxy, c, s in zip(b.xyxy.tolist(), b.cls.tolist(), b.conf.tolist())])
            return out
        preds = detect([frames[r['episode_index']] for r in eps])
        ends = [r for r in eps if r['episode_index'] in lasts]
        preds_end = dict(zip([r['episode_index'] for r in ends], detect([lasts[r['episode_index']] for r in ends])))
        for r, dets in zip(eps, preds):
            ep = r['episode_index']; img = frames[ep]; h, w = img.shape[:2]
            instr = (r.get('tasks') or [''])[0]
            task = task_of(instr)
            phrase, dest = object_phrase(instr, task)
            target = match_class(phrase, names, task)
            def best(cls):
                c = [d for d in dets if d[0] == cls]
                return max(c, key=lambda d: d[5]) if c else None
            dest_cls = f'plate_{dest}' if task == 1 and dest else f'organizer_{dest}' if task == 2 and dest else 'basket' if task == 3 else None
            tb = best(target) if target else None
            db = best(dest_cls) if dest_cls else None
            flags = []
            if task == 0: flags.append('task_unknown')
            if task in (1, 2) and not target: flags.append('target_unparsed')
            if target and not tb: flags.append('target_not_detected')
            if dest_cls and not db: flags.append('dest_not_detected')
            if task == 3:
                balls = [d for d in dets if d[0].startswith('paper_ball')]
                if not balls: flags.append('balls_not_detected')
            key = f"{ds.resolve()}#{ep}"
            t_meta = next((r[c] for c in TIME_COLUMNS if r.get(c)), None)   # a collection time, if the platform stores one
            if key not in seen: seen[key] = stamp
            when = str(t_meta) if t_meta else seen[key]
            rec = {'n': n, 'ds': ds.name, 'ep': ep, 'source': source, 'collector': collector, 'task': task,
                   'instruction': instr, 'target': target, 'dest': dest, 'len': r.get('length'),
                   'dur_s': round((r.get('length') or 0) / fps, 1), 'seen': when, 'time_from': 'meta' if t_meta else 'sync',
                   'dets': dets, 'flags': flags,
                   'target_box': tb[1:5] if tb else None, 'dest_box': db[1:5] if db else None}
            if task == 3:
                rec['balls'] = [d[1:5] for d in dets if d[0].startswith('paper_ball')]
            if ep in sts:
                q = sts[ep]; rec['home_err'] = round(float(np.linalg.norm(q[-1] - q[0])), 4)
                rec['traj'] = traj_metrics(q, fps, task)
                tr = rec['traj']
                if task in (1, 2) and tr.get('regrasp'): flags.append('regrasp')
                if (tr.get('longest_pause') or 0) > 3.0: flags.append('long_pause')
            if ep in preds_end:
                d1 = preds_end[ep]; rec['dets_end'] = d1
                rec['out'], rec['moved'] = outcome(task, target, dest, dets, d1, w)
                if rec['out'] not in ('ok', 'unknown') and (rec.get('traj') or {}).get('home_dist', 0) > 0.5:
                    rec['out'] = 'arm_in_view'   # the arm never went home and covers the scene: the end cannot be judged
                if source != 'rollout':   # a demonstration that does not end as instructed is a bad label
                    flag = {'wrong_plate': 'wrong_place', 'wrong_level': 'wrong_place', 'not_placed': 'not_placed',
                            'wrong_object': 'wrong_object', 'not_moved': 'not_moved', 'partial': 'balls_left',
                            'not_swept': 'balls_left', 'arm_in_view': 'arm_in_view'}.get(rec['out'])
                    if flag: flags.append(flag)
                im1 = Image.fromarray(lasts[ep])
                if w > 320: im1 = im1.resize((320, round(h * 320 / w)))
                im1.save(a.out / 'thumbs' / f'{n}_end.jpg', quality=72)
            if a.gt and (a.gt / f'{ep:04d}' / 'complete.json').exists():
                g = json.loads((a.gt / f'{ep:04d}' / 'complete.json').read_text())
                rec['gt_target_box'] = g.get('target_box')
                gt_stats['checked'] += 1
                sp = g.get('spec', {})
                gt_stats['parse_ok'] = gt_stats.get('parse_ok', 0) + (sp.get('target') == target and str(sp.get('dest')) == str(dest))
                if sp.get('target') != target: gt_stats.setdefault('parse_bad', []).append([instr, sp.get('target'), target])
                if tb and g.get('target_box') and iou(tb[1:5], g['target_box']) >= 0.5: gt_stats['target_hit'] += 1
                elif tb: gt_stats['target_wrong_place'] += 1
                else: gt_stats['target_missed'] += 1
            im = Image.fromarray(img)
            if w > 320: im = im.resize((320, round(h * 320 / w)))
            im.save(a.out / 'thumbs' / f'{n}.jpg', quality=72)
            if task in firsts and len(firsts[task]) < 80: firsts[task].append(img)
            episodes.append(rec); n += 1
    size = None
    for k, imgs in firsts.items():
        if imgs:
            bg = np.median(np.stack(imgs), axis=0).astype(np.uint8)
            Image.fromarray(bg).save(a.out / f'bg_t{k}.jpg', quality=80); size = [bg.shape[1], bg.shape[0]]
    data = {'generated': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), 'title': a.title,
            'image_size': size, 'detector': {'weights': Path(a.weights).name, 'conf': a.conf, 'classes': names},
            'gt_check': gt_stats if a.gt else None, 'backgrounds': {k: f'bg_t{k}.jpg' for k, v in firsts.items() if v},
            'episodes': episodes}
    (a.out / 'field_data.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')))
    if a.seen:
        a.seen.parent.mkdir(parents=True, exist_ok=True); a.seen.write_text(json.dumps(seen))
    print(len(episodes), 'episodes ->', a.out, '| gt', gt_stats if a.gt else '-')


if __name__ == '__main__':
    main()
