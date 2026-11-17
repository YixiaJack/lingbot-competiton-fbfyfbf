"""Merge the official public dataset and team datasets into one LeRobot-style view (camera_top only) for the labelling
scripts: data/merged/ with episode_index offset per dataset, video files symlinked under new file indices, and a
labels.jsonl with synthesized ids (T<task>_<tag>_<n>) for datasets without platform labels.
  make_merged.py public647:0 105a2d09-...:1000 b60f1e2e-...:2000
Public episodes keep their indices, so labels already made with LA_DS=public647 stay valid with LA_DS=merged."""
import json, re, sys
from pathlib import Path
import pandas as pd
H = Path.home() / 'locany'; OUT = H / 'data/merged'; CAM = 'observation.images.camera_top'
def task_of(s):
    s = s.lower()
    if 'plate' in s: return 1
    if any(k in s for k in ('compartment', 'organizer', 'section', 'shelf', 'level')): return 2
    if any(k in s for k in ('paper ball', 'dustpan', 'basket')): return 3
    return 0
(OUT / 'meta/episodes/chunk-000').mkdir(parents=True, exist_ok=True)
(OUT / f'videos/{CAM}/chunk-000').mkdir(parents=True, exist_ok=True)
frames, labels, info0 = [], [], None
for k, arg in enumerate(sys.argv[1:]):
    name, off = arg.rsplit(':', 1); off = int(off); src = H / 'data' / name
    info = json.load(open(src / 'meta/info.json')); info0 = info0 or info
    eps = pd.concat([pd.read_parquet(p) for p in sorted((src / 'meta/episodes').glob('chunk-*/file-*.parquet'))])
    cols = [c for c in eps.columns if not c.startswith('stats/') and (not c.startswith('videos/') or CAM in c)]
    eps = eps[cols].copy()
    fmap = {}
    for (ci, fi) in sorted(set(zip(eps[f'videos/{CAM}/chunk_index'], eps[f'videos/{CAM}/file_index']))):
        nf = 100 * k + len(fmap); fmap[(ci, fi)] = nf
        dst = OUT / f'videos/{CAM}/chunk-000/file-{nf:03d}.mp4'
        if dst.is_symlink() or dst.exists(): dst.unlink()
        dst.symlink_to(src / info['video_path'].format(video_key=CAM, chunk_index=int(ci), file_index=int(fi)))
    eps[f'videos/{CAM}/file_index'] = [fmap[(c, f)] for c, f in zip(eps[f'videos/{CAM}/chunk_index'], eps[f'videos/{CAM}/file_index'])]
    eps[f'videos/{CAM}/chunk_index'] = 0
    lab = {}
    if (src / 'meta/labels.jsonl').exists():
        for l in open(src / 'meta/labels.jsonl'):
            r = json.loads(l); lab[r['episode_index']] = r
    tag = 'pub' if off == 0 else name[:6]
    n_task = {}
    for e, tasks in zip(eps['episode_index'], eps['tasks']):
        e = int(e)
        if e in lab and lab[e]['labels'] and lab[e]['labels'][0].get('custom_fields') and off == 0:
            r = dict(lab[e]); r['episode_index'] = e + off; labels.append(r); continue
        instr = (lab[e]['labels'][0]['task_instruction'] if e in lab and lab[e]['labels'] else None) or list(tasks)[0]
        t = task_of(instr); n_task[t] = n_task.get(t, 0) + 1
        labels.append({'episode_index': e + off, 'labels': [{'task_instruction': instr, 'result': None, 'score': None,
                       'custom_fields': [{'name': 'id', 'value': f'T{t}_{tag}_{n_task[t]:03d}'}]}]})
    eps['episode_index'] = eps['episode_index'] + off
    frames.append(eps)
    print(name, 'episodes', len(eps), 'video files', len(fmap), 'tasks', n_task or 'from platform ids')
allep = pd.concat(frames, ignore_index=True)
allep.to_parquet(OUT / 'meta/episodes/chunk-000/file-000.parquet')
info = dict(info0); info['total_episodes'] = len(allep); info['features'] = {k: v for k, v in info0['features'].items() if not k.startswith('observation.images') or k == CAM}
json.dump(info, open(OUT / 'meta/info.json', 'w'), indent=1)
with open(OUT / 'meta/labels.jsonl', 'w') as f:
    for r in sorted(labels, key=lambda r: r['episode_index']): f.write(json.dumps(r) + '\n')
bad = [r for r in labels if r['labels'][0]['custom_fields'][0]['value'].startswith('T0')]
print('merged', len(allep), 'episodes;', len(bad), 'with unknown task:', [r['labels'][0]['task_instruction'][:60] for r in bad[:5]])
