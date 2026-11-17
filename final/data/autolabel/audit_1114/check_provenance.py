"""Cross-task contamination check for a training list. For every episode of every derived dataset, find its source
episode in the raw data by matching two consecutive observation.state frames (mid-episode) exactly, then compare:
  source task  (public: id prefix T1_/T2_/T3_; own data: raw instruction, official templates first, keywords second)
  prompt task  (the training prompt = meta/tasks.parquet[task_index], classified the same way; T3 subtask prompts -> 3)
  dataset task (from its name: *_t12 -> 1/2, *_t3_sub -> 3, *_t1_box* -> 1, *_t2_box* -> 2)
Writes one JSON line per derived episode to OUT for further checks. Read-only, CPU only.
usage: python check_provenance.py LIST POLICY_JSON OUT SRC_NAME=RAW_DIR ..."""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

lst, policy, out = sys.argv[1], json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
raws = dict(a.split('=', 1) for a in sys.argv[4:])
SUB = {'Sweep all the paper balls into the dustpan with the brush.', 'Lift the dustpan over the basket and pour the paper balls into the basket.',
       'Put the dustpan and the brush down on the table, then return both arms home and stay still.'}
SUB |= {f'Grasp the dustpan handle with the {a} gripper, then grasp the brush handle with the {b} gripper.' for a, b in (('left', 'right'), ('right', 'left'))}
TPL = {}
for k, t in policy['tasks'].items():
    TPL[int(k)] = [re.compile(re.escape(x['text']).replace(r'\{object\}', '(?P<o>.+?)').replace(r'\{destination\}', '(?P<d>.+?)'), re.I)
                   for x in t['templates']]


def ptask(s):
    """(task, how, object, destination) for a prompt"""
    s = s.strip()
    if s in SUB: return 3, 'subtask', None, None
    for k, pats in TPL.items():
        for p in pats:
            m = p.fullmatch(s)
            if m: return k, 'template', m.groupdict().get('o'), m.groupdict().get('d')
    l = s.lower()
    hits = {k for k, ws in ((1, ('plate',)), (2, ('compartment', 'organizer', 'level', 'section', 'tier', 'drawer')),
                            (3, ('dustpan', 'paper ball', 'basket'))) if any(w in l for w in ws)}
    return (hits.pop() if len(hits) == 1 else ('?' if not hits else 'ambiguous:' + ','.join(map(str, sorted(hits))))), 'keyword', None, None


def state_cols(d):
    f = sorted((d / 'data').glob('*/*.parquet'))[0]
    return sorted(c for c in pq.read_schema(f).names if c.startswith('observation.state'))


def load(d, cols):
    eps, S = [], []
    for f in sorted((d / 'data').glob('*/*.parquet')):
        t = pq.read_table(f, columns=['episode_index', 'frame_index'] + cols)
        eps.append(np.stack([t.column('episode_index').to_numpy(), t.column('frame_index').to_numpy()], 1))
        parts = []
        for c in cols:
            a = t.column(c).combine_chunks()
            parts.append(a.flatten().to_numpy(zero_copy_only=False).reshape(len(a), -1) if hasattr(a, 'flatten') else a.to_numpy()[:, None])
        S.append(np.concatenate(parts, 1).astype(np.float32))
    return np.concatenate(eps), np.concatenate(S)


def tasks_of(d):
    t = pd.read_parquet(d / 'meta/tasks.parquet')
    return dict(zip(t['task_index'], t['task'])) if 'task' in t.columns else {int(i): s for s, i in zip(t.index, t['task_index'])}


def labels_of(d):
    L = {}
    p = d / 'meta/labels.jsonl'
    if p.exists():
        for line in open(p):
            r = json.loads(line); lb = r['labels'][0]
            cf = {f['name']: f['value'] for f in (lb.get('custom_fields') or [])}
            L[r['episode_index']] = (lb.get('task_instruction', ''), cf.get('id', ''))
    return L


# index raw frame pairs
cols = None; index = defaultdict(list); src_info = {}
for nm, d in raws.items():
    d = Path(d); c = state_cols(d); cols = c if cols is None else [x for x in cols if x in c]
for nm, d in raws.items():
    d = Path(d); E, S = load(d, cols); L = labels_of(d)
    same = np.r_[E[1:, 0] == E[:-1, 0], False]
    for i in np.nonzero(same)[0]:
        index[hash(S[i].tobytes() + S[i + 1].tobytes())].append((nm, int(E[i, 0]), int(E[i, 1])))
    for e, (ins, i_d) in L.items():
        st = int(i_d[1]) if re.match(r'T[123]_', i_d or '') else ptask(ins)[0]
        src_info[(nm, e)] = dict(instruction=ins, id=i_d, task=st, by='id' if re.match(r'T[123]_', i_d or '') else 'instruction')
    print(f'indexed {nm}: {len(L)} episodes, {len(E)} frames', flush=True)

want = lambda n: {1, 2} if n.endswith('_t12') else {3} if '_t3_sub' in n else {1} if '_t1_box' in n else {2} if '_t2_box' in n else set()
seen = set(); rows = []
for line in open(lst):
    path = Path(line.split()[1])
    if path in seen: continue
    seen.add(path); name = path.name; tk = tasks_of(path); E, S = load(path, cols)
    ti = {}
    for f in sorted((path / 'data').glob('*/*.parquet')):
        t = pq.read_table(f, columns=['episode_index', 'task_index'])
        for e, x in zip(t.column('episode_index').to_pylist(), t.column('task_index').to_pylist()): ti.setdefault(e, set()).add(x)
    for e in np.unique(E[:, 0]):
        idx = np.nonzero(E[:, 0] == e)[0]; m = idx[len(idx) // 2] if len(idx) > 1 else idx[0]
        cands = index.get(hash(S[m].tobytes() + S[m + 1].tobytes()), []) if m + 1 in idx else []
        srcs = sorted({(c[0], c[1]) for c in cands})
        prompt = [tk[x] for x in sorted(ti[int(e)])]
        pt = ptask(prompt[0]); sinfo = [src_info.get(s, {}) for s in srcs]
        stasks = sorted({str(s.get('task')) for s in sinfo})
        rows.append(dict(dataset=name, ep=int(e), frames=len(idx), sources=[f'{a}:{b}' for a, b in srcs], mid_frame_in_src=[c[2] for c in cands][:2],
                         source_task=stasks, source_id=[s.get('id') for s in sinfo], source_instruction=[s.get('instruction') for s in sinfo][:1],
                         prompt=prompt, prompt_task=pt[0], prompt_how=pt[1], prompt_obj=pt[2], prompt_dest=pt[3], dataset_task=sorted(want(name))))
    print(f'checked {name}', flush=True)

out.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows))
print('== per dataset: episodes | no source match | prompt task not allowed | source task != prompt task | several prompts')
for name in dict.fromkeys(r['dataset'] for r in rows):
    R = [r for r in rows if r['dataset'] == name]
    nomatch = [r for r in R if not r['sources']]
    bad_prompt = [r for r in R if r['prompt_task'] not in r['dataset_task']]
    mism = [r for r in R if r['sources'] and r['source_task'] != [str(r['prompt_task'])]]
    multi = [r for r in R if len(r['prompt']) > 1]
    pairs = Counter((','.join(r['source_task']), str(r['prompt_task'])) for r in R)
    print(f"{name:18s} {len(R):4d} | {len(nomatch):3d} | {len(bad_prompt):3d} | {len(mism):3d} | {len(multi):3d}   source->prompt {dict(pairs)}"
          f"   prompt match {dict(Counter(r['prompt_how'] for r in R))}")
    for r in (nomatch + bad_prompt + mism)[:6]:
        print('   !', json.dumps({k: r[k] for k in ('ep', 'sources', 'source_id', 'source_task', 'prompt', 'prompt_task')}, ensure_ascii=False)[:400])
