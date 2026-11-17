"""Task-1 box cues for VLA training from YOLO v1 (LocateAnything labels) on frame 0 of every official demo.

Same cue as the real_detector experiment (cues_real.py): rule-parsed instruction -> target class + plate colour;
white box = target, orange box = destination plate (the one of that colour nearest the target); either box missing ->
no cue for that demo. boxgrasp: the white box is dropped from the frame on which the acting gripper has closed on the
object; the orange box stays (plates never move).

Grasp rule (causal; the robot side applies it online to the gripper commands the VLA outputs): the gripper COMMAND
(action.effector.position) has opened once it exceeds CMD_OPEN, and the object is grasped at the first later frame where
the command falls below CMD_CLOSED. The commanded value is used, not the measured one: on big objects the measured
gripper only drops 0.90 -> 0.83 when it closes on them (ep259, pumpkin), while the command goes 0.90 -> 0.00.
Usage: t1_boxes.py --data <public v1> --dets dets_first.json --out t1_boxes.json [--diffgt ../real_detector/real_check/diffgt.json]
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

CMD_OPEN, CMD_CLOSED = 0.5, 0.2
TEMPLATES = ['Put the {o} on the {d} plate.', 'Pick up the {o} and place it on the {d} plate.',
             'Move the {o} onto the {d} plate.', 'Place the {o} on the {d} plate.', 'Could you put the {o} on the {d} plate?']
# YOLO class -> phrase used in the official instructions (field_training/prompt_policy.json, task 1)
PHRASE = {'wax_apple': 'wax apple', 'bell_pepper': 'bell pepper', 'grapes': 'bunch of grapes', 'tomato': 'tomato',
          'lemon': 'lemon', 'orange': 'orange', 'mangosteen': 'mangosteen', 'persimmon': 'persimmon', 'mushroom': 'mushroom',
          'strawberry': 'strawberry', 'red_delicious': 'red delicious apple', 'tangerine': 'mandarin', 'kiwi': 'kiwi',
          'onion': 'onion', 'avocado': 'avocado', 'pomegranate': 'pomegranate', 'pumpkin': 'pumpkin', 'pear': 'pear',
          'mango': 'mango', 'cherry': 'cherries', 'banana': 'banana', 'carrot': 'carrot', 'red_chili': 'red chili pepper',
          'papaya': 'papaya', 'passion_fruit': 'passion fruit', 'dragon_fruit': 'dragon fruit', 'cucumber': 'cucumber',
          'peach': 'peach', 'watermelon': 'watermelon', 'green_apple': 'green apple', 'pineapple': 'pineapple',
          'eggplant': 'eggplant', 'red_apple': 'red apple', 'potato': 'potato'}
OBJ = {v: k for k, v in PHRASE.items()}
# phrasings seen in our own collected instructions (11/14)
ALIASES = {'cherry': 'cherry', 'grapes': 'grapes', 'kiwi fruit': 'kiwi'}
OBJ.update(ALIASES)
COLORS = ('white', 'yellow', 'pink', 'purple', 'blue')
PATS = [re.compile(re.escape(t).replace(r'\{o\}', '(?P<o>.+?)').replace(r'\{d\}', '(?P<d>[a-z]+)'), re.I) for t in TEMPLATES]


def parse(text):
    for p in PATS:
        m = p.fullmatch(text.strip())
        if m and m['o'].lower() in OBJ and m['d'].lower() in COLORS:
            return OBJ[m['o'].lower()], 'plate_' + m['d'].lower()
    return None


def ctr(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def select(dets, tcls, dcls, conf_target=0.5, conf_dest=0.5, margin=0.15):
    """cues_real.select_boxes on YOLO rows [cls, x1, y1, x2, y2, conf]."""
    t = sorted([(d[5], d[1:5]) for d in dets if d[0] == tcls], key=lambda x: -x[0])
    if not t or t[0][0] < conf_target:
        return None, None, 'target not detected'
    if len(t) > 1 and t[0][0] - t[1][0] < margin:
        return None, None, 'target ambiguous'
    tb = t[0][1]
    d = [(c, b) for c, b in ((d[5], d[1:5]) for d in dets if d[0] == dcls) if c >= conf_dest]
    if not d:
        return None, None, 'destination not detected'
    tc = ctr(tb)
    db = min(d, key=lambda x: (ctr(x[1])[0] - tc[0]) ** 2 + (ctr(x[1])[1] - tc[1]) ** 2)[1]
    return [round(v, 1) for v in tb], [round(v, 1) for v in db], 'ok'


def grasp_frame(cmd):
    """First frame on which some gripper command closes after having opened (None if never)."""
    best = None
    for a in (0, 1):
        opened = False
        for j, v in enumerate(cmd[:, a]):
            if not opened and v > CMD_OPEN:
                opened = True
            elif opened and v < CMD_CLOSED:
                best = j if best is None else min(best, j)
                break
    return best


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True); p.add_argument('--dets', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--diffgt', type=Path)
    a = p.parse_args()
    dets = json.loads(a.dets.read_text())
    lab = {}
    for line in open(a.data / 'meta/labels.jsonl'):
        d = json.loads(line); L = d['labels'][0]
        lab[d['episode_index']] = (next(f['value'] for f in L['custom_fields'] if f['name'] == 'id'), L['task_instruction'])
    eps = sorted(e for e, (i, _) in lab.items() if i.startswith('T1_'))
    t = pq.read_table(sorted((a.data / 'data').glob('*/*.parquet'))[0], columns=['episode_index', 'frame_index', 'action.effector.position'])
    ep_col = t.column('episode_index').to_numpy(); fi_col = t.column('frame_index').to_numpy()
    grc = np.array(t.column('action.effector.position').to_pylist(), dtype=float)
    gt = {int(k): v for k, v in json.loads(a.diffgt.read_text()).items()} if a.diffgt else {}
    out, why, chk = {}, {}, {'checked': 0, 'target_right': 0}
    for e in eps:
        did, instr = lab[e]
        m = ep_col == e; gr = grc[m][np.argsort(fi_col[m])]
        pr = parse(instr)
        if pr is None:
            why['instruction not parsed'] = why.get('instruction not parsed', 0) + 1; continue
        tb, db, reason = select(dets.get(f'first/{e:04d}', []), *pr)
        why[reason] = why.get(reason, 0) + 1
        if reason != 'ok':
            continue
        gf = grasp_frame(gr)
        out[str(e)] = {'id': did, 'instruction': instr, 'target': pr[0], 'dest': pr[1], 'target_box': tb, 'dest_box': db,
                       'grasp_frame': gf, 'T': int(len(gr))}
        g = gt.get(e, {}).get('start_box')
        if g:
            chk['checked'] += 1
            ok = iou(tb, g) >= 0.3 or (g[0] <= ctr(tb)[0] <= g[2] and g[1] <= ctr(tb)[1] <= g[3])
            chk['target_right'] += int(ok); out[str(e)]['target_matches_diffgt'] = bool(ok)
    gfs = [v['grasp_frame'] for v in out.values()]
    summ = {'t1_episodes': len(eps), 'with_cue': len(out), 'reasons': why, 'diffgt_check': chk,
            'grasp_frame_missing': sum(g is None for g in gfs),
            'grasp_frame_s': [round(float(np.percentile([g for g in gfs if g is not None], q)) / 30, 2) for q in (5, 50, 95)],
            'mode': 'boxgrasp', 'line_px': 5, 'colors': {'target': [255, 255, 255], 'dest': [255, 128, 0]}}
    a.out.write_text(json.dumps({'summary': summ, 'episodes': out}, indent=1, ensure_ascii=False))
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == '__main__':
    main()
