"""Task 2 frame 0: single-instance boxes of the whole organizer and of each of the 8 cosmetics, 2 samples each
-> label/cosm0.json {ep: {phrase: [box_sample1, box_sample2]}} (box None if the model gave none)."""
import json, os, re, sys, time
from pathlib import Path
H = Path.home() / 'locany'; M = H / 'models/LocateAnything-3B'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.update(LA_FLASH_MODEL=str(M), LA_FLASH_ATTN='sdpa', LA_FLASH_HYBRID_SCHEDULER='pipeline')
sys.path.insert(0, str(M))
import batch_utils.hybrid_runtime as hr
from batch_utils import generate_batch_hybrid, load
from batch_utils.hybrid_runtime import load_pil
load()
hr._PROMPT = 'Locate a single instance that matches the following description: '
PH = ['clear acrylic organizer', 'beige lip gloss tube', 'beige lotion pump bottle', 'black lipstick with a clear cap',
      'dark-colored round contour compact', 'large black and gold makeup brush', 'light-colored round highlighter compact',
      'pink makeup puff', 'rectangular multicolor eyeshadow palette',
      'small black and gold makeup brush', 'black round glitter compact']   # last two: team data
eps = [json.loads(l)['episode_index'] for l in open(H / 'data' / os.environ.get('LA_DS', 'public647') / 'meta/labels.jsonl')
       if json.loads(l)['labels'][0]['custom_fields'][0]['value'].startswith('T2')]
prev = json.load(open(H / 'label/cosm0.json')) if (H / 'label/cosm0.json').exists() else {}
eps = [e for e in eps if str(e) not in prev]
ims = {e: load_pil(H / f'label/first0/images/ep{e:03d}/0000.jpg') for e in eps}
B = re.compile(r'<box><(\d+)><(\d+)><(\d+)><(\d+)></box>')
import torch
jobs = [(e, p) for p in PH for e in eps] * 2
part = H / 'label/cosm0.partial.json'
out = {int(k): v for k, v in json.load(open(part)).items()} if part.exists() else {}
out.update({int(k): v for k, v in prev.items()}); n_prev = sum(len(v) for d in prev.values() for v in d.values())
done = sum(len(v) for d in out.values() for v in d.values()) - n_prev
jobs = jobs[done:]   # jobs are processed in order, so the first `done` are finished
t0 = time.time()
def gen(pairs):
    try:
        return generate_batch_hybrid(pairs, temperature=0.7, top_p=0.9, top_k=None, repetition_penalty=1.1,
                                     max_new_tokens=256, scheduler='pipeline', group_size=0)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(pairs) == 1:
            return ['']
        h = len(pairs) // 2
        return gen(pairs[:h]) + gen(pairs[h:])
for i in range(0, len(jobs), 4):
    ch = jobs[i:i + 4]
    texts = gen([(ims[e], p) for e, p in ch])
    if i % 200 == 0:
        torch.cuda.empty_cache(); json.dump(out, open(part, 'w'))
    for (e, p), tx in zip(ch, texts):
        b = B.findall(tx)
        out.setdefault(e, {}).setdefault(p, []).append([int(v) * s / 1000 for v, s in zip(b[0], (640, 480, 640, 480))] if b else None)
    if i % 400 == 0:
        print(f'{i}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
json.dump(out, open(H / 'label/cosm0.json', 'w'))
print(f'done {len(jobs)} queries in {time.time() - t0:.0f}s')
