"""Build batch 22 (b60f1e2e 2c3d609e, 19:00), eps 486-510 (no task-2 normal demos in this batch):
  own_t2_dagger_2c3d   task-2 DAgger 494-510: HUMAN TAKEOVER SEGMENTS ONLY (dagger_seg), checked by eye, official-template prompts;
                       500 = policy only (no human part); 501 dropped (the highlighter ends on the front lip of the bottom level,
                       half outside, labelled partial); 509_0 dropped (3.6 s, the arm just holds the bottle, then the policy resumes)
  own_t3_sub_new2c3d   task-3 demos 486-493 x 4 subtasks (real_split; 489 with open_thr 0.34; 490 = demo: flagged by the jerk rule
                       only, no policy gripper frames, human throughout), checked by eye"""
import json, random, re, sys, collections
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import yolocue_bridge as YC, derive_dataset as DD
V1 = Path.home() / 'Downloads/b60f1e2e_2c3d609e/v1'; O = Path('out/ds_own2c3d'); O.mkdir(parents=True, exist_ok=True); fps = 30
seg = json.load(open('out/dagger_2c3d.json'))
DROP = {'501_0': 'the highlighter ends on the front lip of the bottom level, half outside (labelled partial)',
        '509_0': '3.6 s human segment: the arm only holds the bottle, then the policy resumes'}
clean = lambda s: re.sub(r'\b(onion|mangosteen) fruit\b', r'\1', re.sub(r'\bstar fruit\b', 'starfruit', s, flags=re.I), flags=re.I)
pol = json.load(open(Path.home() / '.cache/field_dev/watch/repo/prompt_policy.json'))['tasks']
OWN = {'starfruit': 'starfruit', 'green_chili': 'green chili pepper', 'glitter_compact': 'black round glitter compact', 'brush_small': 'small black and gold makeup brush'}
def allocate(n, weights, seed):
    tot = sum(weights); raw = [n * w / tot for w in weights]; cnt = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - cnt[i], reverse=True)[:n - sum(cnt)]: cnt[i] += 1
    seq = [i for i, c in enumerate(cnt) for _ in range(c)]; random.Random(seed).shuffle(seq); return seq
keep = {k: dict(v) for k, v in seg.items() if k not in DROP}
by = collections.defaultdict(list)
for k, v in keep.items(): pr = YC.parse(clean(v['instruction'])); assert pr, k; by[pr[0]].append((k, pr))
for t, lst in by.items():
    TT = pol[str(t)]; seq = allocate(len(lst), [x['weight'] for x in TT['templates']], f'dagger2c3d:{t}')
    for (k, pr), i in zip(sorted(lst), seq):
        obj = TT['object_phrases'].get(pr[1]) or OWN[pr[1]]
        txt = TT['templates'][i]['text'].format(object=obj, destination=pr[2].split('_', 1)[1]); assert YC.parse(txt) == pr, (k, txt)
        keep[k]['train_instruction'] = txt
final = {'dropped': DROP, 'segments': dict(sorted(keep.items()))}
json.dump(final, open(O / 'dagger_2c3d_final.json', 'w'), indent=1, ensure_ascii=False)
for k, v in final['segments'].items(): print(f"  {k} {v['start']}-{v['end']} {v['human_s']}s {v['result']} | {v['train_instruction']}")
items = [{'src_ep': v['ep'], 'start': v['start'], 'end': v['end'], 'task': v['train_instruction']} for v in final['segments'].values()]
print('own_t2_dagger_2c3d', DD.build(V1, O / 'own_t2_dagger_2c3d', items, fps))
P = json.load(open('out/t3_2c3d.json'))
it = [{'src_ep': v['ep'], 'start': v['start'], 'end': v['end'], 'task': v['task'], 'subtask': v['subtask'], 'pan_arm': v['pan_arm'], 'kind': v['kind']}
      for v in sorted(P.values(), key=lambda v: (v['ep'], v['start']))]
print('own_t3_sub_new2c3d', DD.build(V1, O / 'own_t3_sub_new2c3d', it, fps), dict(collections.Counter(v['subtask'] for v in P.values())))
json.dump({'note': 'batch 22 task 3; checked by eye 2026-11-15 19:20', 'pieces': P}, open(O / 't3_2c3d_final.json', 'w'), indent=1, ensure_ascii=False)
