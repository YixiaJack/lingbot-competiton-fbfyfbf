"""Batch 24 = b60f1e2e c89c4a87 (22:22), eps 533-543, all task 3 (classify_c89c.py decides demo vs DAgger).
Normal demos (dagger_seg None) -> the 4 subtasks of real_split.segment, whole episode.
DAgger -> human takeover segments only, cut at the subtask boundaries; user rule (11/15): the subtask the human took over IN
= takeover piece (DAgger weight, own_t3_dagger_c89c_to); the following subtasks the human completed = follow-up pieces
(normal demo weight, own_t3_sub_newc89c with the normal demos). Pieces < 1 s dropped.
Instructions: real_split.instruction. Writes out/t3_c89c.json and out/chk_t3_c89c_N.jpg."""
import json, sys
from pathlib import Path
import av, numpy as np, pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import dagger_seg as DS, real_split as R
V1 = Path.home() / 'Downloads/b60f1e2e_c89c4a87/v1'; CAM = 'observation.images.camera_top'; FPS = 30
meta = {m['episode_index']: m for m in pq.read_table(sorted((V1 / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()}
lab = {}
for l in open(V1 / 'meta/labels.jsonl'):
    r = json.loads(l); L = (r.get('labels') or [{}])[0]; lab[r['episode_index']] = (L.get('task_instruction') or '', L.get('result'))
cols = ['episode_index', 'frame_index', 'action.effector.position', 'action.arm.position', 'observation.state.arm.position',
        'observation.state.effector.position', 'observation.state.ee.position']
t = pq.read_table(sorted((V1 / 'data').glob('*/*.parquet'))[0], columns=cols); ep, fi = t.column('episode_index').to_numpy(), t.column('frame_index').to_numpy()
C, A, S, G, E = (np.array(t.column(c).to_pylist(), float) for c in cols[2:])


def frames(e, idx):
    m = meta[e]; p = V1 / 'videos' / CAM / f"chunk-{m[f'videos/{CAM}/chunk_index']:03d}" / f"file-{m[f'videos/{CAM}/file_index']:03d}.mp4"; t0 = m[f'videos/{CAM}/from_timestamp']
    want = sorted(set(idx)); ts = [t0 + i / FPS for i in want]; got = {}; k = 0
    with av.open(str(p)) as ic:
        s = ic.streams.video[0]; ic.seek(int(max(t0 + want[0] / FPS - 1, 0) / s.time_base), stream=s, backward=True)
        for fr in ic.decode(s):
            while k < len(want) and float(fr.pts * s.time_base) + 1e-3 >= ts[k]: got[want[k]] = fr.to_image(); k += 1
            if k == len(want): break
    return got


out = {}; cells = []; F = ImageFont.load_default(size=13)
for e in range(533, 544):
    m = np.where(ep == e)[0]; m = m[np.argsort(fi[m])]; T = len(m)
    segs, _, _ = DS.segments(C[m], A[m], S[m]); sp = R.segment(G[m], E[m]); pan = sp['pan_arm']
    b = [0, sp['b1'], sp['b2'], sp['b3'], T]
    spans = [(0, T)] if segs is None else segs
    for si, (st, en) in enumerate(spans):
        first = True
        for k, sub in enumerate(R.SUBTASK_IDS):
            a, z = max(st, b[k]), min(en, b[k + 1])
            if z - a < FPS: continue
            kind = 'demo' if segs is None else ('takeover' if first else 'follow')
            first = False
            out[f'{e}_{sub}_{a}'] = {'ep': e, 'start': int(a), 'end': int(z), 'subtask': sub, 'pan_arm': pan, 'kind': kind,
                                     'secs': round((z - a) / FPS, 1), 'task': R.instruction(sub, pan), 'result': lab[e][1]}
    pcs = [v for v in out.values() if v['ep'] == e]
    print(f"ep{e} T{T} {'DAgger ' + str(segs) if segs else 'demo'} pan {pan} b {b} | " + ', '.join(f"{v['subtask']}:{v['kind']} {v['start']}-{v['end']}" for v in pcs))
    idx = ([max(segs[0][0] - 45, 0), segs[0][0]] if segs else [0]) + [x for bb in b[1:4] for x in (bb - 1, bb)] + [T - 3]
    idx = [i for i in idx if i < T][:10]
    fr = frames(e, idx); cell = Image.new('RGB', (10 * 192, 172), 'white')
    for k, i in enumerate(idx):
        sub = next(R.SUBTASK_IDS[j] for j in range(4) if b[j] <= i < b[j + 1])
        cell.paste(fr[i].resize((192, 144)), (k * 192, 28)); ImageDraw.Draw(cell).text((k * 192 + 3, 14), f'f{i} {sub}', font=F, fill=(160, 0, 0))
    ImageDraw.Draw(cell).text((4, 0), f"ep{e} ({lab[e][1]}) {'DAgger human ' + str(segs) if segs else 'normal demo'} | pan {pan} | b1 {b[1]} b2 {b[2]} b3 {b[3]} T {T}", font=F, fill='black')
    cells.append(cell)
for k in range(0, len(cells), 6):
    cc = cells[k:k + 6]; sh = Image.new('RGB', (1920, 172 * len(cc)), 'white')
    for i, c_ in enumerate(cc): sh.paste(c_, (0, 172 * i))
    sh.save(f'out/chk_t3_c89c_{k // 6 + 1}.jpg', quality=85)
json.dump(out, open('out/t3_c89c.json', 'w'), indent=1, ensure_ascii=False)
import collections
print({k: dict(collections.Counter(v['subtask'] for v in out.values() if v['kind'] == k)) for k in ('demo', 'takeover', 'follow')})
