"""Task 1 (fruit_id.py, generated from cosm_id.py): identify every fruit/vegetable on frame 0 by appearance.
Exemplars = confirmed target start boxes (main frame-0 box agreeing with ground0, train episodes only), labelled with the
instruction's class. Candidates = stable single-instance boxes (cosm0.json, both samples IoU >= 0.5) plus the all-category
boxes (first0_t2), merged. Each candidate gets the kNN class (ResNet-50 features, cosine, exemplars from other episodes).
Prints leave-one-episode-out accuracy on the exemplars, writes label/fruit_id.json {ep: [[cls, box, vote_share]]}."""
import json, os, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, torch, torchvision
from PIL import Image
H = Path.home() / 'locany'; L = H / 'label'
T1_OBJ = ['avocado', 'banana', 'bell_pepper', 'carrot', 'cucumber', 'dragon_fruit', 'eggplant', 'green_apple', 'kiwi',
          'lemon', 'mango', 'mangosteen', 'mushroom', 'onion', 'orange', 'papaya', 'passion_fruit', 'peach', 'pear',
          'persimmon', 'pineapple', 'pomegranate', 'potato', 'pumpkin', 'red_apple', 'strawberry', 'tomato', 'watermelon', 'wax_apple']
C2 = {c.replace('_', ' '): c for c in T1_OBJ}
C2.update({'bunch of grapes': 'grapes', 'cherries': 'cherry', 'mandarin': 'tangerine', 'red chili pepper': 'red_chili',
           'red delicious apple': 'red_delicious'})
C2.update({'kiwi fruit': 'kiwi', 'grapes': 'grapes', 'cherry': 'cherry', 'starfruit': 'starfruit', 'green chili pepper': 'green_chili'})
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)
meta = {}
for l in open(H / 'data' / os.environ.get('LA_DS', 'public647') / 'meta/labels.jsonl'):
    r = json.loads(l); lab = r['labels'][0]; cid = lab['custom_fields'][0]['value']
    if cid.startswith('T1'):
        ph = re.search(r'(?:Pick up|[Pp]ut|Place|Move|[Cc]ould you put) the (.+?)(?: and place it)? (?:on|onto) the \w+ plate', lab['task_instruction'])
        ph = ph.group(1) if ph else None
        if ph in C2: meta[r['episode_index']] = (C2[ph], 'reserve' in cid)
g0 = {int(k): v for k, v in json.load(open(L / 'ground0.json')).items()}
cos = {}
start, last = {}, {}
for l in open(L / 'main/labels.jsonl'):
    r = json.loads(l)
    if r['task'] != 1 or not r.get('target_box'): continue
    if r['frame'] == 0: start[r['ep']] = r['target_box']
    if r['level'] == 0 and (r['ep'] not in last or r['frame'] > last[r['ep']][0]): last[r['ep']] = (r['frame'], r['target_box'])
multi = {r['ep']: r['boxes'] for r in map(json.loads, open(L / 'first0/labels.jsonl')) if r['task'] == 1}

w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
net = torchvision.models.resnet50(weights=w).eval(); net.fc = torch.nn.Identity()
tf = w.transforms()
def crops(e, boxes, frame=0):
    p = L / f'main/images/ep{e:03d}/{frame:04d}.jpg'
    im = Image.open(p if p.exists() else L / f'first0/images/ep{e:03d}/0000.jpg').convert('RGB'); out = []
    for b in boxes:
        cw, ch = b[2] - b[0], b[3] - b[1]; s = max(cw, ch) * 0.6 + 8; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        out.append(tf(im.crop([int(cx - s), int(cy - s), int(cx + s), int(cy + s)])))
    return out
@torch.no_grad()
def emb(ts):
    f = net(torch.stack(ts)); return torch.nn.functional.normalize(f, dim=1).numpy()

ex_e, ex_c, ex_t = [], [], []
for e, (cls, res) in meta.items():
    if res or e not in start or not g0.get(e, {}).get('tgt') or iou(start[e], g0[e]['tgt'][0]) < 0.5: continue
    ex_e.append(e); ex_c.append(cls); ex_t += crops(e, [start[e]])
    if e in last:   # second view: the target lying on the plate at the end
        ex_e.append(e); ex_c.append(cls); ex_t += crops(e, [last[e][1]], last[e][0])
EX = emb(ex_t); ex_e = np.array(ex_e); ex_c = np.array(ex_c)
def knn(f, not_ep, k=7):
    m = ex_e != not_ep; sims = EX[m] @ f; o = np.argsort(-sims)[:k]
    v = Counter(ex_c[m][o]); c, n = v.most_common(1)[0]; return c, n / k
ACCEPT = 5 / 7 - 1e-6
# exemplars are target boxes that are ~81% right: drop the ones their neighbours confidently call something else
pred = [knn(EX[i], ex_e[i]) for i in range(len(ex_e))]
keep = np.array([not (c != ex_c[i] and sh >= ACCEPT) for i, (c, sh) in enumerate(pred)])
print('exemplars', len(ex_e), 'dropped as likely mislabelled', int((~keep).sum()))
bad_target = [int(e) for e in ex_e[~keep]]
EX, ex_e, ex_c = EX[keep], ex_e[keep], ex_c[keep]
acc = Counter()
for i in range(len(ex_e)):
    c, sh = knn(EX[i], ex_e[i]); acc[(ex_c[i], c == ex_c[i], sh >= ACCEPT)] += 1
n_acc = sum(v for (k, ok, a_), v in acc.items() if a_)
print('LOO accuracy all', round(sum(v for (k, ok, a_), v in acc.items() if ok) / len(ex_e), 3),
      '| accepted (>=5/7)', n_acc, 'accuracy', round(sum(v for (k, ok, a_), v in acc.items() if ok and a_) / max(1, n_acc), 3))
for cls in sorted(set(ex_c)):
    t = sum(v for (k, ok, a_), v in acc.items() if k == cls and a_); r = sum(v for (k, ok, a_), v in acc.items() if k == cls and a_ and ok)
    print(f'  {cls:18s} accepted {t}, right {r}')

out = {}; nk = Counter()
for e, (tcls, res) in meta.items():   # reserve targets were not exemplars: check them the same way
    if res and e in start:
        c, sh = knn(emb(crops(e, [start[e]]))[0], e)
        if c != tcls and sh >= ACCEPT: bad_target.append(e)
print('episodes whose target box looks like another cosmetic:', len(bad_target))
for e, (tcls, res) in meta.items():
    cand = []
    for ph, s in cos.get(e, {}).items():
        if ph in C2 and len(s) == 2 and all(s) and iou(s[0], s[1]) >= 0.5: cand.append(s[0])
    cand += [x[1:] for x in multi.get(e, []) if not x[0].startswith('plate_')]
    plates = [x[1:] for x in multi.get(e, []) if x[0].startswith('plate_')]
    org = None
    objs = []
    for b in cand:
        if (b[2] - b[0]) * (b[3] - b[1]) > 0.06 * 640 * 480: continue
        if any(iou(b, p) >= 0.5 for p in plates): continue
        if org and org[0] <= (b[0] + b[2]) / 2 <= org[2] and org[1] <= (b[1] + b[3]) / 2 <= org[3]: continue
        if e in start and iou(b, start[e]) >= 0.3: continue      # the target itself comes from the start box
        if any(iou(b, o) >= 0.3 for o in objs): continue
        objs.append(b)
    if not objs: out[e] = []; continue
    F = emb(crops(e, objs)); best = {}
    for b, f in zip(objs, F):
        c, s = knn(f, e)
        if s < ACCEPT or c == tcls: nk['rejected'] += 1; continue      # >= 5 of 7 neighbours; the target class is already used
        if c not in best or s > best[c][1]: best[c] = (b, s)
    out[e] = [[c, b, s] for c, (b, s) in best.items()]; nk['kept'] += len(out[e])
out['_bad_target_eps'] = bad_target
json.dump(out, open(L / 'fruit_id.json', 'w'))
print(dict(nk), 'per episode', round(nk['kept'] / len(meta), 2), Counter(c for k, v in out.items() if k != '_bad_target_eps' for c, _, _ in v))
