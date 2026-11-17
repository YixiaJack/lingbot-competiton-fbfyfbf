"""v13 evaluation (eval12 + the yolo13 hold-out: own eps 418/234/231/316/32 and today's live raw frames 175700/172549/173022) on real-robot data only (by-eye labels, never trained on):
  - cue right/wrong on the yolo8 (16), yolo9 (12), yolo10-own (2) and yolo11 (26) hold-outs, guard on/off
  - per object on the yolo11 hold-out: found / confused-as / missed, focus on confusable classes
  - cross-class overlaps (two classes >= 0.3 on one object) on the yolo11 hold-out
usage: python -I eval11.py MODEL [MODEL ...]"""
import json, sys, collections, re
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0, '.')
import yolocue_bridge as YC, v7_boxes as VB, yc_guard as G
area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
same = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1])) / max(min(area(a), area(b)), 1) > 0.5
V = {'old': Path.home() / 'Downloads/b60f1e2e_0d3bf9f6/v1', 'new': Path.home() / 'Downloads/b60f1e2e_3541a365/v1', 'f64d': Path.home() / 'Downloads/b60f1e2e_f64d6d11/v1'}
META = {k: {m['episode_index']: m for m in pq.read_table(sorted((v / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()} for k, v in V.items()}
ins = {}
for f in ('instr_all.json', 'instr_0200.json', 'instr_0308.json'):
    for k, v in json.load(open(f)).items(): ins[int(k)] = v
for l in open(V['f64d'] / 'meta/labels.jsonl'):
    r = json.loads(l); L = (r.get('labels') or [{}])[0]
    if r['episode_index'] >= 302: ins[r['episode_index']] = re.sub(r'\b(onion|mangosteen) fruit\b', r'\1', L.get('task_instruction') or '').replace('the apple', 'the red apple')
SETS = [('yolo8/ds/final_labels.json', 'old'), ('yolo9/ds/final_labels.json', 'old'), ('yolo10/ds_own/final_labels.json', 'old'), ('yolo11/ds/final_labels.json', 'new'), ('yolo12/ds/final_labels.json', 'f64d')]
frames = {}
for lf, vk in SETS:
    FL = json.load(open(lf))
    for e in FL['holdout']: frames[e] = (VB.frame0(V[vk], META[vk][e]), FL['labels'][str(e)], lf.split('/')[0])
CONF = ['mangosteen', 'grapes', 'starfruit', 'lemon', 'tomato', 'wax_apple', 'lip_gloss', 'lotion_bottle', 'red_chili', 'carrot', 'starfruit', 'brush_small', 'brush_large', 'powder_puff', 'potato', 'persimmon', 'bell_pepper', 'cucumber', 'green_chili']
# yolo13 hold-out: own episodes (77235417 snapshot, frame 0) + live raw frames; a cue is right when it lands on the labelled
# target (and destination); when the instruction names an object that is not in the scene (172549/173022 ask for a lemon), "none" is right
V13 = Path.home() / 'Downloads/b60f1e2e_77235417/v1'; M13 = {m['episode_index']: m for m in pq.read_table(sorted((V13 / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()}
P13 = json.load(open('yolo13/prop/props.json')); F13 = json.load(open('yolo13/ds/final_labels.json')); L13 = json.load(open('yolo13/ds/live_labels.json')); LP = json.load(open('yolo13/live_prop/props.json'))
import numpy as _np
from PIL import Image as _Im
for e in F13['holdout']:
    rgb = _np.asarray(_Im.open(f'yolo13/prop/frames/ep{e}_0000.jpg').convert('RGB')); frames[f'o{e}'] = (rgb, F13['labels'][str(e)], 'yolo13'); ins[f'o{e}'] = P13[str(e)]['instruction']
for b in L13['labels']:
    if b[:15] in L13['holdout']:
        rgb = _np.asarray(_Im.open(f'yolo13/live/{b}_nocue_raw.jpg').convert('RGB')); frames[f'l{b[9:15]}'] = (rgb, L13['labels'][b], 'yolo13'); ins[f'l{b[9:15]}'] = LP[b]['task']
CONF += ['pear', 'red_apple', 'red_delicious', 'pomegranate', 'mango', 'papaya']
for n in sys.argv[1:]:
    det = YC.detector({**YC.DEFAULTS, 'weights': f'weights/{n}.torchscript', 'names': f'weights/{n}.names.json', 'device': 'cpu'})
    cue = collections.defaultdict(lambda: collections.Counter()); obj = collections.defaultdict(collections.Counter); conf = collections.Counter(); overl = collections.Counter(); wrongs = []
    for e, (rgb, truth, setn) in sorted(frames.items(), key=lambda kv: str(kv[0])):
        pr = YC.parse(ins[e]); d = det(rgb); L = {}
        for k, b in truth: L.setdefault(k, b)
        for tag, fn in (('plain', YC.select), ('guard', G.select_guarded)):
            tb, db, why = fn(d, *pr, rgb=rgb)
            if tb:
                ok = pr[1] in L and same(tb, L[pr[1]]) and (pr[2] not in L or same(db, L[pr[2]])); cue[(setn, tag)]['right' if ok else 'wrong'] += 1
                if not ok and tag == 'plain': wrongs.append((e, pr[1], why))
            else: cue[(setn, tag)]['none (right: target absent)' if pr[1] not in L else 'none'] += 1
        if setn not in ('yolo11', 'yolo12', 'yolo13'): continue
        hi = [x for x in d if x[5] >= 0.5]
        for k, b in truth:
            if k.startswith(('plate_', 'organizer_')): continue
            s_ = [x for x in hi if x[0] == k and same(x[1:5], b)]; o_ = [x for x in hi if x[0] != k and same(x[1:5], b)]
            if s_ and not o_: obj[k]['found'] += 1
            elif o_: obj[k]['confused'] += 1; conf[(k, max(o_, key=lambda x: x[5])[0])] += 1
            else: obj[k]['missed'] += 1
        lo = [x for x in d if x[5] >= 0.3 and not x[0].startswith(('plate_', 'organizer_'))]
        for i in range(len(lo)):
            for j in range(i + 1, len(lo)):
                if lo[i][0] != lo[j][0] and G.iou(lo[i][1:5], lo[j][1:5]) > 0.5: overl[tuple(sorted((lo[i][0], lo[j][0])))] += 1
    print(f'===== {n}')
    for k in sorted(cue): print('  cue', k, dict(cue[k]))
    print('  wrong cues (plain):', wrongs)
    tot = collections.Counter(); [tot.update(v) for v in obj.values()]; print('  yolo11+12+13 hold-out objects:', dict(tot))
    print('  confusable classes:', {k: dict(obj[k]) for k in CONF if k in obj})
    print('  yolo13 hold-out per object:', [(e, [(k, (lambda s_: '%s %.2f' % (max(s_)[1], max(s_)[0]) if s_ else None)([(x[5], x[0]) for x in det(frames[e][0]) if x[5] >= 0.3 and same(x[1:5], bb)])) for k, bb in frames[e][1] if not k.startswith(('plate_', 'organizer_'))]) for e in frames if frames[e][2] == 'yolo13'])
    print('  confusions (true->pred):', conf.most_common(10)); print('  cross-class overlaps >=0.3:', dict(overl))
