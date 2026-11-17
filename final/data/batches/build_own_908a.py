"""Build the b60f1e2e 908aa6e5 (16:05) task-2 datasets, same recipe as b6ef:
  own_t12_new908a         17 task-2 demos (447-465 minus 454 DAgger and 462 dropped), whole episodes, official-template prompts
  own_t12_box_v12_new908a the same 17 with YOLO v12 box cues (out/own908a_cues.json, checked by eye)
  own_t2_dagger_908a      439-446, 454 DAgger recordings: HUMAN TAKEOVER SEGMENTS ONLY (dagger_seg), checked by eye;
                          446 dropped (the gripper stays closed through the whole human segment: the compact is still held when
                          it ends; the release happens after it). Task-3 DAgger 466-474: separate package (t3 subtask split).
Box dataset: run derive_box_dataset.py afterwards (see __main__ note)."""
import json, random, re, sys, collections
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import yolocue_bridge as YC, derive_dataset as DD
V1 = Path.home() / 'Downloads/b60f1e2e_908aa6e5/v1'; O = Path('out/ds_own908a'); O.mkdir(parents=True, exist_ok=True)
fps = json.loads((V1 / 'meta/info.json').read_text())['fps']
instr = {int(k): v for k, v in json.load(open('instr_908a.json')).items()}
_, EPS, _ = DD.load_src(V1)
items = [{'src_ep': e, 'start': 0, 'end': EPS[e]['length'], 'task': instr[e]} for e in sorted(instr)]
print('own_t12_new908a', DD.build(V1, O / 'own_t12_new908a', items, fps))

# DAgger: final segments + official-template prompts (same allocation as the plain set, own seed)
seg = json.load(open('out/dagger_908a.json'))
DROP = {'446_0': 'gripper command stays closed (0.0) through the whole human segment 569-1345: the compact is still held when it ends, the release comes after'}
CUT = {}
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
    TT = pol[str(t)]; seq = allocate(len(lst), [x['weight'] for x in TT['templates']], f'dagger908a:{t}')
    for (k, pr), i in zip(sorted(lst), seq):
        obj = TT['object_phrases'].get(pr[1]) or OWN[pr[1]]
        txt = TT['templates'][i]['text'].format(object=obj, destination=pr[2].split('_', 1)[1]); assert YC.parse(txt) == pr, (k, txt)
        keep[k]['train_instruction'] = txt
final = {'note': '446 dropped: ' + DROP['446_0'], 'segments': dict(sorted(keep.items()))}
json.dump(final, open('out/dagger_908a_final.json', 'w'), indent=1, ensure_ascii=False)
for k, v in final['segments'].items(): print(f"  {k} {v['start']}-{v['end']} {v['human_s']}s {v['result']} | {v['instruction']} -> {v['train_instruction']}")
items = [{'src_ep': v['ep'], 'start': v['start'], 'end': v['end'], 'task': v['train_instruction']} for v in final['segments'].values()]
print('own_t2_dagger_908a', DD.build(V1, O / 'own_t2_dagger_908a', items, fps))
