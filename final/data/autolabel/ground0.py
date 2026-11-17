"""Single-instance grounding of the instructed target and destination on frame 0 (T1/T2), for checking vs diffgt."""
import json, os, re, sys, time
from pathlib import Path
H = Path.home() / 'locany'; M = H / 'models/LocateAnything-3B'
os.environ.update(LA_FLASH_MODEL=str(M), LA_FLASH_ATTN='sdpa', LA_FLASH_HYBRID_SCHEDULER='pipeline')
sys.path.insert(0, str(M))
import batch_utils.hybrid_runtime as hr
from batch_utils import generate_batch_hybrid, load
from batch_utils.hybrid_runtime import load_pil
load()
hr._PROMPT = 'Locate a single instance that matches the following description: '
rows = []
for line in open(H / 'data' / os.environ.get('LA_DS', 'public647') / 'meta/labels.jsonl'):
    r = json.loads(line); lab = r['labels'][0]; cid = lab['custom_fields'][0]['value']; s = lab['task_instruction']
    t = int(cid[1])
    if t == 1:
        m = re.search(r'(?:Pick up|[Pp]ut|Place|Move|[Cc]ould you put) the (.+?)(?: and place it)? (?:on|onto) the (\w+) plate', s)
        if not m: continue
        tgt, dest = m.group(1), f'{m.group(2)} plate'
    elif t == 2:
        m = re.search(r'the (.+?)(?: and put it)? (?:in|into|inside) the (top|middle|bottom)', s)
        if not m: continue
        tgt, dest = m.group(1), f'{m.group(2)} shelf of the organizer'
    else:
        continue
    rows.append((r['episode_index'], t, tgt, dest))
prev = json.load(open(H / 'label/ground0.json')) if (H / 'label/ground0.json').exists() else {}
rows = [r for r in rows if str(r[0]) not in prev]
out = {int(k): v for k, v in prev.items()}
t0 = time.time()
for kind in ('tgt', 'dest'):
    for i in range(0, len(rows), 8):
        ch = rows[i:i + 8]
        pairs = [(load_pil(H / f'label/first0/images/ep{e:03d}/0000.jpg'), tg if kind == 'tgt' else de) for e, t, tg, de in ch]
        texts = generate_batch_hybrid(pairs, temperature=0.7, top_p=0.9, top_k=None, repetition_penalty=1.1,
                                      max_new_tokens=256, scheduler='pipeline', group_size=0)
        for (e, t, tg, de), tx in zip(ch, texts):
            b = re.findall(r'<box><(\d+)><(\d+)><(\d+)><(\d+)></box>', tx)
            out.setdefault(e, {'task': t, 'tgt_phrase': tg, 'dest_phrase': de})[kind] = [[int(v) * s / 1000 for v, s in zip(x, (640, 480, 640, 480))] for x in b]
            out[e][kind + '_raw'] = tx
print(f'{len(rows)} episodes x2 in {time.time() - t0:.0f}s')
json.dump(out, open(H / 'label/ground0.json', 'w'))
