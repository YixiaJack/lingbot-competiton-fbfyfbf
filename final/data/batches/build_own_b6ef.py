"""Build the b60f1e2e b6ef6674 (14:01) datasets, eps 412-438, same recipe as f64d:
  own_t12_newb6ef         412-422 normal teleop demos, whole episodes, official-template prompts (instr_b6ef.json)
  own_t12_box_v12_newb6ef 9 of them with YOLO v12 box cues (out/ownb6ef_cues.json, checked by eye; 414/418 pear not detected)
  own_t2_dagger_b6ef      423-438 DAgger recordings: HUMAN TAKEOVER SEGMENTS ONLY (dagger_seg), checked by eye;
                          424 dropped (human segment ends with the bottle still in the gripper, arm moving; the rest is policy),
                          432 end cut to 1875 and 438 end cut to 800 (a human hand enters the frame at f1881-1890 / f805-817)
Box dataset: run derive_box_dataset.py afterwards (see __main__ note)."""
import json, random, re, sys, collections
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import yolocue_bridge as YC, derive_dataset as DD
V1 = Path.home() / 'Downloads/b60f1e2e_b6ef6674/v1'; O = Path('out/ds_ownb6ef'); O.mkdir(parents=True, exist_ok=True)
fps = json.loads((V1 / 'meta/info.json').read_text())['fps']
instr = {int(k): v for k, v in json.load(open('instr_b6ef.json')).items()}
_, EPS, _ = DD.load_src(V1)
items = [{'src_ep': e, 'start': 0, 'end': EPS[e]['length'], 'task': instr[e]} for e in sorted(instr)]
print('own_t12_newb6ef', DD.build(V1, O / 'own_t12_newb6ef', items, fps))

# DAgger: final segments + official-template prompts (same allocation as the plain set, own seed)
seg = json.load(open('out/dagger_b6ef.json'))
DROP = {'424_0': 'human segment ends at f815 with the bottle still in the gripper and the arm moving (ee step 0.0087); the rest of the episode is policy'}
CUT = {'432_0': (1875, 'end cut to 1875: a human hand enters at the right edge between f1881 and f1890'),
       '438_0': (800, 'end cut to 800: a human hand reaches into the organizer from f805-817')}
clean = lambda s: re.sub(r'\b(onion|mangosteen) fruit\b', r'\1', re.sub(r'\bstar fruit\b', 'starfruit', s, flags=re.I), flags=re.I)
pol = json.load(open(Path.home() / '.cache/field_dev/watch/repo/prompt_policy.json'))['tasks']
OWN = {'starfruit': 'starfruit', 'green_chili': 'green chili pepper', 'glitter_compact': 'black round glitter compact', 'brush_small': 'small black and gold makeup brush'}
def allocate(n, weights, seed):
    tot = sum(weights); raw = [n * w / tot for w in weights]; cnt = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - cnt[i], reverse=True)[:n - sum(cnt)]: cnt[i] += 1
    seq = [i for i, c in enumerate(cnt) for _ in range(c)]; random.Random(seed).shuffle(seq); return seq
keep = {k: dict(v) for k, v in seg.items() if k not in DROP}
for k, (en, why) in CUT.items(): keep[k]['end'] = en; keep[k]['note'] = why; keep[k]['human_s'] = round((en - keep[k]['start']) / 30, 1)
by = collections.defaultdict(list)
for k, v in keep.items(): pr = YC.parse(clean(v['instruction'])); assert pr, k; by[pr[0]].append((k, pr))
for t, lst in by.items():
    TT = pol[str(t)]; seq = allocate(len(lst), [x['weight'] for x in TT['templates']], f'daggerb6ef:{t}')
    for (k, pr), i in zip(sorted(lst), seq):
        obj = TT['object_phrases'].get(pr[1]) or OWN[pr[1]]
        txt = TT['templates'][i]['text'].format(object=obj, destination=pr[2].split('_', 1)[1]); assert YC.parse(txt) == pr, (k, txt)
        keep[k]['train_instruction'] = txt
final = {'note': '424 dropped: ' + DROP['424_0'], 'segments': dict(sorted(keep.items()))}
json.dump(final, open('out/dagger_b6ef_final.json', 'w'), indent=1, ensure_ascii=False)
for k, v in final['segments'].items(): print(f"  {k} {v['start']}-{v['end']} {v['human_s']}s {v['result']} | {v['instruction']} -> {v['train_instruction']}")
items = [{'src_ep': v['ep'], 'start': v['start'], 'end': v['end'], 'task': v['train_instruction']} for v in final['segments'].values()]
print('own_t2_dagger_b6ef', DD.build(V1, O / 'own_t2_dagger_b6ef', items, fps))
