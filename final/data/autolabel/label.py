"""Auto-label camera_top frames of the official 647 real demos with LocateAnything-3B (YOLO training data).

  label.py --name first0 --frames first          # frame 0 of every episode (quality check vs real_check diffgt)
  label.py --name main --strides 30,10,5,1       # progressive: j%30==0 for all episodes, then j%10, j%5, every frame

Tasks 1/2: only the instructed target moves, so each frame gets a single-instance grounding of the instruction's
target phrase (frame-0 check vs diffgt: 94% / 81% right, against 67% / 58% for the all-categories prompt); the
destination and the other objects come from frame 0 (ground0.json, first0*/labels.jsonl). Task 3: every frame gets the
all-categories prompt (5 distinct classes).
Output OUT=~/locany/label/<name>/: labels.jsonl (one row per frame, append-only, resumable),
images/ep{e:03d}/{j:04d}.jpg (q95), progress.json.  Box rows: [class, x1, y1, x2, y2] in pixels.
Yields the GPU: exits with code 3 if OUT/../PAUSE exists or another process opens a CUDA context (run.sh waits).
"""
import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
from decord import VideoReader, cpu
from PIL import Image

HOME = Path.home() / 'locany'
DS = HOME / 'data' / os.environ.get('LA_DS', 'public647')
MODEL = HOME / 'models/LocateAnything-3B'
CAM = 'observation.images.camera_top'
PAUSE = HOME / 'label' / 'PAUSE'

# prompt phrase -> class key (keys follow experiments/real_detector catalog names; new: paper_ball_yellow)
T1 = {p: p.replace(' ', '_') for p in [
    'avocado', 'banana', 'bell pepper', 'carrot', 'cucumber', 'dragon fruit', 'eggplant', 'green apple', 'kiwi',
    'lemon', 'mango', 'mangosteen', 'mushroom', 'onion', 'orange', 'papaya', 'passion fruit', 'peach', 'pear',
    'persimmon', 'pineapple', 'pomegranate', 'potato', 'pumpkin', 'red apple', 'strawberry', 'tomato',
    'watermelon', 'wax apple', 'starfruit']}
T1.update({'bunch of grapes': 'grapes', 'cherries': 'cherry', 'mandarin': 'tangerine',
           'red chili pepper': 'red_chili', 'red delicious apple': 'red_delicious', 'green chili pepper': 'green_chili'})
T1.update({f'{c} plate': f'plate_{c}' for c in ['white', 'yellow', 'pink', 'purple', 'blue']})
# LocateAnything emits names in 6-token blocks: long phrases get cut ("bottom compartment of" + "the clear organizer"),
# so task-2 prompts are kept short (11/14)
T2 = {'lip gloss': 'lip_gloss', 'lotion bottle': 'lotion_bottle', 'lipstick': 'lipstick',
      'dark round compact': 'contour', 'makeup brush': 'brush_large', 'light round compact': 'highlighter',
      'pink powder puff': 'powder_puff', 'eyeshadow palette': 'eyeshadow_palette',
      'small makeup brush': 'brush_small', 'glitter compact': 'glitter_compact'}   # last two: team data 11/14
T2.update({f'{lv} shelf': f'organizer_{lv}' for lv in ['top', 'middle', 'bottom']})
T3 = {'brush': 'brush', 'dustpan': 'dustpan', 'basket': 'basket',
      'red paper ball': 'paper_ball_red', 'yellow paper ball': 'paper_ball_yellow'}
VOCAB = {1: T1, 2: T2, 3: T3}
QUERY = {t: '</c>'.join(v) for t, v in VOCAB.items()}
REF = re.compile(r'<ref>(.*?)</ref>((?:<box>(?:<\d+>)+</box>)*)')
BOX = re.compile(r'<box><(\d+)><(\d+)><(\d+)><(\d+)></box>')


def parse(raw, task, w, h):
    out = []
    for name, boxes in REF.findall(raw):
        cls = VOCAB[task].get(name.strip(), '?' + name.strip())
        for b in BOX.findall(boxes):
            x1, y1, x2, y2 = (int(v) for v in b)
            out.append([cls, round(x1 * w / 1000, 1), round(y1 * h / 1000, 1), round(x2 * w / 1000, 1), round(y2 * h / 1000, 1)])
    return out


def other_gpu_procs():
    try:
        r = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=20)
        return [int(p) for p in r.stdout.split() if p.strip().isdigit() and int(p) != os.getpid()]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--frames', choices=['first', 'strides'], default='strides')
    ap.add_argument('--strides', default='30,10,5,1')
    ap.add_argument('--tasks', default='1,2,3')
    ap.add_argument('--batch-size', type=int, default=6)
    a = ap.parse_args()
    out = HOME / 'label' / a.name
    (out / 'images').mkdir(parents=True, exist_ok=True)

    info = json.load(open(DS / 'meta/info.json')); fps = info['fps']
    eps = pd.read_parquet(DS / 'meta/episodes/chunk-000/file-000.parquet').set_index('episode_index')
    task_of, target_of = {}, {}
    for line in open(DS / 'meta/labels.jsonl'):
        r = json.loads(line); lab = r['labels'][0]; cid = lab['custom_fields'][0]['value']; ins = lab['task_instruction']
        e = r['episode_index']; task_of[e] = int(cid[1])
        m_ = None
        if task_of[e] == 1:
            m_ = re.search(r'(?:Pick up|[Pp]ut|Place|Move|[Cc]ould you put) the (.+?)(?: and place it)? (?:on|onto) the \w+ plate', ins)
        elif task_of[e] == 2:
            m_ = re.search(r'the (.+?)(?: and put it)? (?:in|into|inside) the (?:top|middle|bottom)', ins)
        if m_:
            target_of[e] = m_.group(1)
        elif task_of[e] in (1, 2) and a.frames != 'first':
            print(f'ep{e}: cannot parse the target from {ins!r}; skipped', flush=True); task_of[e] = 0
    tasks = {int(t) for t in a.tasks.split(',')}
    by_task = {t: [e for e in sorted(task_of) if task_of[e] == t] for t in sorted(tasks)}
    order = [e for grp in zip(*by_task.values()) for e in grp]          # round-robin over tasks
    order += [e for t in by_task for e in by_task[t] if e not in set(order)]

    done = set()
    lab = out / 'labels.jsonl'
    if lab.exists():
        for line in open(lab):
            try:
                r = json.loads(line); done.add((r['ep'], r['frame']))
            except json.JSONDecodeError:
                pass   # torn last line after a kill
    plan = []   # (level, ep, [frames])
    if a.frames == 'first':
        plan = [(0, e, [0]) for e in sorted(order, key=lambda e: task_of[e])]
    else:
        strides = [int(s) for s in a.strides.split(',')]
        for lv, s in enumerate(strides):
            for e in order:
                n = int(eps.loc[e, 'length'])
                fr = [j for j in range(0, n, s) if all(j % p for p in strides[:lv])]
                if fr:
                    plan.append((lv, e, fr))
    total = sum(len(f) for _, _, f in plan)
    todo = [(lv, e, [j for j in fr if (e, j) not in done]) for lv, e, fr in plan]
    todo = [x for x in todo if x[2]]
    print(f'plan {total} frames, done {len(done)}, todo {sum(len(f) for *_, f in todo)}', flush=True)

    # 11/14: reserved memory crept up over ~2000 frames until OOM (varying shapes fragment the caching allocator)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.update(LA_FLASH_MODEL=str(MODEL), LA_FLASH_ATTN='sdpa', LA_FLASH_HYBRID_SCHEDULER='pipeline')
    sys.path.insert(0, str(MODEL))
    import torch
    from batch_utils import generate_batch_hybrid, load
    load()

    q = queue.Queue(maxsize=6)
    stop = threading.Event()

    flat = [(lv, e, j) for lv, e, fr in todo for j in fr]
    batches, cur = [], []
    for it in flat:
        if cur and (len(cur) == a.batch_size or (cur[-1][0], task_of[cur[-1][1]]) != (it[0], task_of[it[1]])):
            batches.append(cur); cur = []
        cur.append(it)
    if cur:
        batches.append(cur)

    def producer():
        vrs = {}
        for k_b, b in enumerate(batches):
            if k_b % 200 == 199:   # decord leaks host memory on random-access get_batch: reopen the readers
                vrs.clear(); import gc; gc.collect()
            ims = []
            for lv, e, j in b:
                r = eps.loc[e]; fi = int(r[f'videos/{CAM}/file_index'])
                if fi not in vrs:
                    vrs[fi] = VideoReader(str(DS / info['video_path'].format(video_key=CAM, chunk_index=0, file_index=fi)), ctx=cpu(0))
                base = int(round(r[f'videos/{CAM}/from_timestamp'] * fps))
                ims.append((e, j, base + j, fi))
            # one get_batch per (file) keeps decord seeks cheap for consecutive frames
            pil = {}
            for fi in {x[3] for x in ims}:
                sel = [x for x in ims if x[3] == fi]
                arr = vrs[fi].get_batch([x[2] for x in sel]).asnumpy()
                for x, a_ in zip(sel, arr):
                    pil[(x[0], x[1])] = Image.fromarray(a_)
            out_ims = []
            for lv, e, j in b:
                im = pil[(e, j)]; d = out / 'images' / f'ep{e:03d}'; d.mkdir(exist_ok=True)
                p = d / f'{j:04d}.jpg'
                if not p.exists():
                    im.save(p, quality=95)
                out_ims.append(im)
            while not stop.is_set():
                try:
                    q.put((b, out_ims), timeout=1); break
                except queue.Full:
                    pass
            if stop.is_set():
                return
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    import batch_utils.hybrid_runtime as hr
    MULTI = hr._PROMPT
    SINGLE = 'Locate a single instance that matches the following description: '

    def gen(pairs, single=False):
        hr._PROMPT = SINGLE if single else MULTI
        try:
            return generate_batch_hybrid(pairs, temperature=0.7, top_p=0.9, top_k=None, repetition_penalty=1.1,
                                         max_new_tokens=256 if single else 1024, scheduler='pipeline', group_size=0)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(pairs) == 1:
                return ['<oom>']   # skip this frame (bad='no_end') instead of killing the run
            h = len(pairs) // 2
            return gen(pairs[:h], single) + gen(pairs[h:], single)

    def bad(raw, task, single=False):
        if single:
            if not raw.rstrip().endswith('<|im_end|>'):
                return 'no_end'
            return 'n_box' if len(BOX.findall(raw)) > 1 else None
        # greedy decoding loops on these prompts (11/14); sampling occasionally truncates or garbles a ref
        if not raw.rstrip().endswith('<|im_end|>'):
            return 'no_end'
        refs = REF.findall(raw)
        names = [n.strip() for n, _ in refs]
        if any(n not in VOCAB[task] for n in names):
            return 'unknown_ref'
        if any(len(BOX.findall(bx)) > 25 for _, bx in refs):
            return 'too_many'
        if len(re.findall(r'<ref>', raw)) != len(refs):
            return 'malformed'
        return None

    def infer(pairs, task, tries=3):
        single = task in (1, 2) and a.frames != 'first'
        texts = gen(pairs, single); why = [bad(t, task, single) for t in texts]; ntry = [1] * len(texts)
        for k in range(1, tries):
            redo = [i for i, w in enumerate(why) if w]
            if not redo:
                break
            again = gen([pairs[i] for i in redo], single)
            for i, t in zip(redo, again):
                texts[i] = t; why[i] = bad(t, task, single); ntry[i] += 1
        return texts, why, ntry

    f = open(lab, 'a')
    t0 = time.time(); n = 0; nb = 0; per_lv = {}
    while True:
        item = q.get()
        if item is None:
            break
        b, ims = item
        t = task_of[b[0][1]]
        single = t in (1, 2) and a.frames != 'first'
        texts, why, ntry = infer([(im, target_of[e] if single else QUERY[t]) for (_, e, _), im in zip(b, ims)], t)
        for (lv, e, j), im, raw, w_, k in zip(b, ims, texts, why, ntry):
            w, h = im.size
            if single:   # class comes from the instruction, not from the model's ref text
                bx = [[None, round(int(x1) * w / 1000, 1), round(int(y1) * h / 1000, 1), round(int(x2) * w / 1000, 1),
                       round(int(y2) * h / 1000, 1)] for x1, y1, x2, y2 in BOX.findall(raw)] if not w_ else []
                row = {'target': target_of[e], 'target_box': bx[0][1:] if bx else None}
            else:
                row = {'boxes': parse(raw, t, w, h) if not w_ else []}
            f.write(json.dumps({'ep': e, 'frame': j, 'task': t, 'level': lv, 'w': w, 'h': h, 'tries': k, 'bad': w_,
                                **row, 'raw': raw}) + '\n')
        f.flush()
        lv, e, js = b[-1][0], b[-1][1], [x[2] for x in b]
        n += len(js); nb += 1; per_lv[lv] = per_lv.get(lv, 0) + len(js)
        if nb % 50 == 0:
            torch.cuda.empty_cache()
        if nb % 25 == 0:
            import psutil
            rss = psutil.Process().memory_info().rss / 2**30
            if rss > 7:   # 11/14 15:35: label.py reached 29 GB RSS and, with a CPU job beside it, hung the machine
                print(f'RSS {rss:.1f} GB > 7 GB: restarting to free host memory', flush=True)
                stop.set(); f.close(); sys.exit(3)
        if nb % 25 == 0:
            el = time.time() - t0
            (out / 'progress.json').write_text(json.dumps({
                'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'done_before': len(done), 'done_now': n,
                'plan': total, 'fps': round(n / el, 2), 'eta_h': round((total - len(done) - n) / (n / el) / 3600, 2),
                'last': [e, js[-1]], 'per_level_now': per_lv,
                'mem_reserved_gb': round(torch.cuda.memory_reserved() / 2**30, 2)}))
            print(f'{n} frames {n / el:.2f} f/s  last ep{e} f{js[-1]}  reserved {torch.cuda.memory_reserved() / 2**30:.1f} GB', flush=True)
        if (PAUSE.exists() and not os.environ.get('LA_IGNORE_PAUSE')) or (nb % 2 == 0 and other_gpu_procs()):
            print('yield GPU: ' + ('PAUSE file' if PAUSE.exists() else f'other CUDA procs {other_gpu_procs()}'), flush=True)
            stop.set(); f.close(); sys.exit(3)
    f.close()
    (out / 'DONE').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + '\n')
    print(f'done {n} frames in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
