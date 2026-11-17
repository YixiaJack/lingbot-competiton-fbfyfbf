import sys, json, glob
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'code'))
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
import field_build as fb
ds = Path(sys.argv[1]); out = Path(sys.argv[2]); (out/'first').mkdir(parents=True, exist_ok=True); (out/'last').mkdir(exist_ok=True)
cam = 'observation.images.camera_top'
rows = []
for f in sorted(glob.glob(str(ds/'meta/episodes/*/*.parquet'))):
    rows += [{k: v for k, v in r.items() if 'stats' not in k} for r in pq.read_table(f).to_pylist()]
labels = {json.loads(l)['episode_index']: json.loads(l)['labels'][0] for l in open(ds/'meta/labels.jsonl')}
idx = []
for where in ('first', 'last'):
    fr = fb.frames_at(ds, cam, rows, where)
    for ep, img in fr.items():
        Image.fromarray(img).save(out/where/f'{ep:04d}.png')
    print(where, len(fr), flush=True)
for r in rows:
    lab = labels[r['episode_index']]
    idx.append({'ep': r['episode_index'], 'instr': r['tasks'][0], 'len': r['length'],
                'id': next(f['value'] for f in lab['custom_fields'] if f['name'] == 'id'), 'result': lab['result']})
(out/'index.json').write_text(json.dumps(idx, ensure_ascii=False))
