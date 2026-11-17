"""Frame-0 single-instance boxes of the three organizer levels for every task-2 episode -> label/levels0.json."""
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
eps = [json.loads(l)['episode_index'] for l in open(H / 'data/public647/meta/labels.jsonl')
       if json.loads(l)['labels'][0]['custom_fields'][0]['value'].startswith('T2')]
B = re.compile(r'<box><(\d+)><(\d+)><(\d+)><(\d+)></box>')
out, t0 = {}, time.time()
for lv in ('top', 'middle', 'bottom'):
    for i in range(0, len(eps), 8):
        ch = eps[i:i + 8]
        pairs = [(load_pil(H / f'label/first0/images/ep{e:03d}/0000.jpg'), f'{lv} shelf of the organizer') for e in ch]
        for _ in range(3):   # sampling; retry rows with no box
            texts = generate_batch_hybrid(pairs, temperature=0.7, top_p=0.9, top_k=None, repetition_penalty=1.1,
                                          max_new_tokens=256, scheduler='pipeline', group_size=0)
            for e, tx in zip(ch, texts):
                b = B.findall(tx)
                if b and lv not in out.setdefault(e, {}):
                    out[e][lv] = [int(v) * s / 1000 for v, s in zip(b[0], (640, 480, 640, 480))]
            pairs = [p for e, p in zip(ch, pairs) if lv not in out.get(e, {})]; ch = [e for e in ch if lv not in out.get(e, {})]
            if not ch:
                break
json.dump(out, open(H / 'label/levels0.json', 'w'))
print(f'{len(out)} episodes, {sum(len(v) == 3 for v in out.values())} with all 3 levels, {time.time() - t0:.0f}s')
