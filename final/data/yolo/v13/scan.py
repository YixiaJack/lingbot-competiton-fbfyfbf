"""YOLO v13 candidate scan: frame 0 of every task-1 episode in own b60f1e2e (77235417 meta + local videos) and 105a2d09
41c236b9 (policy runs), v12 at conf 0.05. Keeps scenes where any focus class (pear, tomato, pomegranate + look-alikes potato,
red_delicious, red_apple, persimmon, peach) is detected >= 0.10 or named in the instruction. Writes yolo13/scan.json."""
import json, glob, re, sys
from pathlib import Path
import av, numpy as np, pyarrow.parquet as pq
sys.path.insert(0, str(Path.home() / 'lingbot_v7work')); import yolocue_bridge as YC
FOCUS = {'pear', 'tomato', 'pomegranate', 'potato', 'red_delicious', 'red_apple', 'persimmon', 'peach'}
W = re.compile(r'\b(pears?|tomato(es)?|pomegranates?|potato(es)?|red delicious|apple)\b', re.I)
det = YC.Detector(YC.ROOT / 'weights/v12.torchscript', json.loads((YC.ROOT / 'weights/v12.names.json').read_text())['names'], conf=0.05)
C = 'observation.images.camera_top'
SRC = {'own': (Path.home() / 'Downloads/b60f1e2e_77235417/v1', [Path.home() / f'Downloads/{d}/v1' for d in ('b60f1e2e_77235417', 'b60f1e2e_3541a365', 'b60f1e2e_0d3bf9f6')]),
       'pol': (Path.home() / 'Downloads/105a2d09_41c236b9/v1', [Path.home() / 'Downloads/105a2d09_41c236b9/v1'])}
out = {}
for src, (V, vids) in SRC.items():
    meta = {m['episode_index']: m for m in pq.read_table(sorted((V / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()}
    lab = {}
    for l in open(V / 'meta/labels.jsonl'):
        r = json.loads(l); L = (r.get('labels') or [{}])[0]; lab[r['episode_index']] = L.get('task_instruction') or ''
    byfile = {}
    for e, m in meta.items():
        p = YC.parse(re.sub(r'\bstar fruit\b', 'starfruit', lab.get(e, ''), flags=re.I))
        if not p or p[0] != 1: continue
        k = (m[f'videos/{C}/chunk_index'], m[f'videos/{C}/file_index'])
        byfile.setdefault(k, []).append(e)
    for k, eps in sorted(byfile.items()):
        f = next((d / 'videos' / C / f'chunk-{k[0]:03d}' / f'file-{k[1]:03d}.mp4' for d in vids if (d / 'videos' / C / f'chunk-{k[0]:03d}' / f'file-{k[1]:03d}.mp4').exists()), None)
        if f is None: print(src, 'missing video', k, len(eps)); continue
        eps = sorted(eps, key=lambda e: meta[e][f'videos/{C}/from_timestamp'])
        with av.open(str(f)) as ic:
            s = ic.streams.video[0]
            for e in eps:
                t0 = meta[e][f'videos/{C}/from_timestamp']; ic.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s, backward=True)
                rgb = next((fr.to_ndarray(format='rgb24') for fr in ic.decode(s) if float(fr.pts * s.time_base) + 1e-3 >= t0), None)
                if rgb is None: print(src, 'ep', e, 'not in', f.parent.parent.parent.parent.parent.name); continue
                d = det(rgb); foc = [(x[0], round(x[5], 2)) for x in d if x[0] in FOCUS and x[5] >= 0.10]
                named = bool(W.search(lab[e]))
                if foc or named:
                    out[f'{src}_{e}'] = {'src': src, 'ep': e, 'instruction': lab[e], 'named': named, 'focus': sorted(foc, key=lambda x: -x[1])[:8]}
    print(src, 'task-1 eps', sum(len(v) for v in byfile.values()), '-> candidates', sum(1 for v in out.values() if v['src'] == src))
json.dump(out, open(Path.home() / 'lingbot_v7work/yolo13/scan.json', 'w'), indent=1, ensure_ascii=False)
