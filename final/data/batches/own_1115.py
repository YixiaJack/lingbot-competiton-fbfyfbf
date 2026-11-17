"""Own data (b60f1e2e cbd7865e): instructions for the box datasets and the new plain task-1/2 set.
  old episodes (already in v4n): keep the instruction their training copy uses (rewritten to official phrasing on 11/14)
  new episodes: rewrite to the official templates in the official proportions (largest remainder, seeded shuffle),
                object phrase = the official one (prompt_policy object_phrases) or our own name for the 4 team objects
writes instr_all.json {ep: instruction} (all task-1/2), new_t12.json {ep: instruction}, new_t3.json [eps]"""
import json, random, re, sys, collections
from pathlib import Path
sys.path.insert(0, '.')
import yolocue_bridge as YC
V1 = Path.home() / 'Downloads/b60f1e2e_cbd7865e/v1'
M = {int(k): v for k, v in json.load(open('own_map_cbd7.json')).items()}
pol = json.load(open(Path.home() / '.cache/field_dev/watch/repo/prompt_policy.json'))['tasks']
OWN = {'starfruit': 'starfruit', 'green_chili': 'green chili pepper', 'glitter_compact': 'black round glitter compact', 'brush_small': 'small black and gold makeup brush'}
lab = {}
for l in open(V1 / 'meta/labels.jsonl'):
    r = json.loads(l); L = (r.get('labels') or [{}])[0]; lab[r['episode_index']] = (L.get('task_instruction') or '', {c.get('name'): c.get('value') for c in (L.get('custom_fields') or [])})
def task_of(s):
    s = s.lower(); return 3 if 'dustpan' in s else 1 if 'plate' in s else 2 if any(w in s for w in ('compartment', 'section', 'level', 'organizer')) else 0
def allocate(n, weights, seed):
    tot = sum(weights); raw = [n * w / tot for w in weights]; cnt = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - cnt[i], reverse=True)[:n - sum(cnt)]: cnt[i] += 1
    seq = [i for i, c in enumerate(cnt) for _ in range(c)]; random.Random(seed).shuffle(seq); return seq
instr_all, new_t12, new_t3, problems = {}, {}, [], []
newby = collections.defaultdict(list)
for e, (s, cf) in sorted(lab.items()):
    t = task_of(s)
    if t == 3:
        if not M.get(e): new_t3.append(e)
        continue
    if t not in (1, 2): problems.append((e, s, 'task?')); continue
    if M.get(e):
        instr_all[e] = next(v for v in M[e].values() if v); continue
    pr = YC.parse(s)
    if pr is None:   # loose object word: use the custom field
        for v in cf.values():
            for task, pats, table, dests, prefix in YC._PATS:
                if str(v).lower() in table: pr = (t, table[str(v).lower()], None)
        if pr is None: problems.append((e, s, 'not parsed')); continue
        m = re.search(r'(white|yellow|pink|purple|blue|top|middle|bottom)', s.lower()); pr = (pr[0], pr[1], ('plate_' if t == 1 else 'organizer_') + m.group(1))
    newby[t].append((e, pr))
for t in (1, 2):
    T = pol[str(t)]; seq = allocate(len(newby[t]), [x['weight'] for x in T['templates']], f'own1115:{t}')
    for (e, pr), i in zip(newby[t], seq):
        obj = T['object_phrases'].get(pr[1]) or OWN.get(pr[1]); dest = pr[2].split('_', 1)[1]
        if obj is None: problems.append((e, lab[e][0], f'no phrase for {pr[1]}')); continue
        txt = T['templates'][i]['text'].format(object=obj, destination=dest)
        assert YC.parse(txt) == pr, (txt, YC.parse(txt), pr)
        instr_all[e] = new_t12[e] = txt
json.dump({str(k): v for k, v in instr_all.items()}, open('instr_all.json', 'w'), ensure_ascii=False, indent=1)
json.dump({str(k): v for k, v in new_t12.items()}, open('new_t12.json', 'w'), ensure_ascii=False, indent=1)
json.dump(new_t3, open('new_t3.json', 'w'))
print('task-1/2 instructions', len(instr_all), '| new task-1/2', len(new_t12), '| new task-3', len(new_t3), new_t3)
for e, s in new_t12.items(): print(f'  {e}: {lab[e][0][:60]!r} -> {s!r}')
print('problems', problems)
c = collections.Counter((YC.parse(s)[0], next(p for p in ('Put the', 'Pick up the', 'Move the', 'Place the', 'Could you put', 'Store the', 'Take the', 'Organize the') if s.startswith(p))) for s in instr_all.values())
print('own task-1/2 template counts after:', sorted(c.items()))
