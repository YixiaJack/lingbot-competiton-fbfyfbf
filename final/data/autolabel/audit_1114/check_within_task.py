"""Within-task checks on the provenance rows (prov_v3.jsonl from check_provenance.py):
 A. box datasets: matched source episode == source_map src_episode, prompt == source_map task, box target/dest classes ==
    what the prompt names (t1_boxes.parse / task-2 template parse), and word overlap between target class and prompt object.
 B. own-data prompts (rewritten into official phrasing): destination (plate colour / organizer level) equals the raw
    instruction's, object words overlap; lists every distinct (raw instruction -> prompt) object pair.
 C. public prompts equal the official raw instruction.
usage: python check_within_task.py PROV POLICY SRCMAPS_DIR..."""
import json, re, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/autolabel'))
import t1_boxes as TB

rows = [json.loads(l) for l in open(sys.argv[1])]
pol = json.loads(Path(sys.argv[2]).read_text())['tasks']
smaps = {}
for d in sys.argv[3:]:
    for p in Path(d).glob('*/meta/source_map.json'):
        smaps[p.parent.parent.name] = {r['episode_index']: r for r in json.loads(p.read_text())}
OBJ2 = {v.lower(): k for k, v in pol['2']['object_phrases'].items()}
P2 = [re.compile(re.escape(t['text']).replace(r'\{object\}', '(?P<o>.+?)').replace(r'\{destination\}', '(?P<d>top|middle|bottom)'), re.I) for t in pol['2']['templates']]
COL = ('white', 'yellow', 'pink', 'purple', 'blue', 'green', 'red', 'orange', 'black', 'brown', 'gray', 'grey')
LEV = ('top', 'middle', 'bottom', 'upper', 'lower', 'first', 'second', 'third')


def parse2(s):
    for p in P2:
        m = p.fullmatch(s.strip())
        if m and m['o'].lower() in OBJ2: return OBJ2[m['o'].lower()], 'organizer_' + m['d'].lower()


def words(s):
    return set(re.findall(r'[a-z]+', (s or '').lower().replace('_', ' '))) - {'the', 'a', 'an', 'of', 'and', 'with', 'small', 'large', 'plate', 'organizer'}


def dest_of(s, task):
    l = (s or '').lower()
    if task == 1:
        c = [x for x in COL if re.search(rf'\b{x}\b(?= plate)', l)] or [x for x in COL if re.search(rf'\b{x}\b', l)]
        return c[-1] if c else None
    lv = [x for x in LEV if re.search(rf'\b{x}\b', l)]
    norm = {'upper': 'top', 'first': 'top', 'lower': 'bottom', 'third': 'bottom', 'second': 'middle'}
    return norm.get(lv[0], lv[0]) if lv else None


print('== A. box datasets')
for ds, sm in smaps.items():
    if '_box' not in ds: continue
    R = [r for r in rows if r['dataset'] == ds]
    bad = Counter(); ex = []
    for r in R:
        m = sm.get(r['ep'])
        src_eps = {int(s.split(':')[1]) for s in r['sources']}
        task = 1 if '_t1_' in ds else 2
        par = TB.parse(r['prompt'][0]) if task == 1 else parse2(r['prompt'][0])
        checks = {'no source_map row': m is None,
                  'source episode differs': m is not None and m['src_episode'] not in src_eps,
                  'prompt differs from box instruction': m is not None and m['task'] != r['prompt'][0],
                  'box classes != prompt': m is not None and (par is None or (m['target'], m['dest']) != tuple(par)),
                  'target class shares no word with prompt object': m is not None and not (words(m['target']) & words(r['prompt'][0]))}
        for k, v in checks.items():
            if v: bad[k] += 1; ex.append((k, r['ep'], r['prompt'][0], m and (m['target'], m['dest'])))
    print(f'{ds:18s} {len(R):4d} episodes, source_map rows {len(sm)}, problems {dict(bad) or "none"}')
    for e in ex[:6]: print('   !', e)

print('== B/C. prompt vs raw instruction (object and destination)')
for ds in dict.fromkeys(r['dataset'] for r in rows):
    if '_t3_sub' in ds: continue
    R = [r for r in rows if r['dataset'] == ds]
    exact = sum(r['prompt'][0] == (r['source_instruction'] or [''])[0] for r in R)
    dbad, obad, pairs = [], [], Counter()
    for r in R:
        t = r['prompt_task']; raw = (r['source_instruction'] or [''])[0]; pr = r['prompt'][0]
        dr, dp = dest_of(raw, t), dest_of(pr, t)
        if dr != dp: dbad.append((r['ep'], raw, pr))
        if raw != pr:
            po = r['prompt_obj'] or pr
            pairs[(raw, po)] += 1
            if not (words(raw) & words(po)): obad.append((r['ep'], raw, pr))
    print(f'{ds:18s} {len(R):4d} | prompt == raw instruction {exact} | destination differs {len(dbad)} | object shares no word {len(obad)}')
    for e in (dbad + obad)[:8]: print('   !', e)
    if ds.startswith('priv'):
        for (raw, po), n in sorted(pairs.items()):
            print(f'     {n}x  "{raw}"  ->  object "{po}"')
