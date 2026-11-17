"""New task-1 episodes of b60f1e2e 53e3655c (02:00): official-distribution prompt rewrite, v7a cues, datasets."""
import json, random, re, subprocess, sys, collections
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import yolocue_bridge as YC, derive_dataset as DD
V1 = Path.home() / 'Downloads/b60f1e2e_53e3655c/v1'; new = json.load(open('new0200_eps.json'))
pol = json.load(open(Path.home() / '.cache/field_dev/watch/repo/prompt_policy.json'))['tasks']
OWN = {'starfruit': 'starfruit', 'green_chili': 'green chili pepper', 'glitter_compact': 'black round glitter compact', 'brush_small': 'small black and gold makeup brush'}
lab = {}
for l in open(V1 / 'meta/labels.jsonl'):
    r = json.loads(l); lab[r['episode_index']] = ((r.get('labels') or [{}])[0].get('task_instruction') or '')
def allocate(n, weights, seed):
    tot = sum(weights); raw = [n * w / tot for w in weights]; cnt = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - cnt[i], reverse=True)[:n - sum(cnt)]: cnt[i] += 1
    seq = [i for i, c in enumerate(cnt) for _ in range(c)]; random.Random(seed).shuffle(seq); return seq
by = collections.defaultdict(list)
for e in new:
    s = re.sub(r'\bstar fruit\b', 'starfruit', lab[e], flags=re.I); pr = YC.parse(s)
    assert pr, (e, lab[e]); by[pr[0]].append((e, pr))
instr = {}
for t, lst in by.items():
    T = pol[str(t)]; seq = allocate(len(lst), [x['weight'] for x in T['templates']], f'own0200:{t}')
    for (e, pr), i in zip(lst, seq):
        obj = T['object_phrases'].get(pr[1]) or OWN[pr[1]]
        txt = T['templates'][i]['text'].format(object=obj, destination=pr[2].split('_', 1)[1]); assert YC.parse(txt) == pr; instr[e] = txt
        print(f'  {e}: {lab[e]!r} -> {txt!r}')
json.dump({str(k): v for k, v in instr.items()}, open('instr_0200.json', 'w'), ensure_ascii=False, indent=1)
Path('instr_0200_dbd.json').write_text(json.dumps({'episodes': {str(k): {'instruction': v} for k, v in instr.items()}}, ensure_ascii=False))
print(collections.Counter(next(p for p in ('Put the', 'Pick up the', 'Move the', 'Place the', 'Could you put') if s.startswith(p)) for s in instr.values()))
