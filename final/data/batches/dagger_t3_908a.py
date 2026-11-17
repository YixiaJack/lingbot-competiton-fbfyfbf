"""Task-3 DAgger recordings in b60f1e2e 908aa6e5 (466-474, harness sessions with a human takeover).
Human takeover segments by dagger_seg.segments (as for task 1/2); the subtask boundaries of the whole episode by
real_split.segment (grasp [0,b1), sweep [b1,b2), dump [b2,b3), home [b3,T)); each human segment is cut at those
boundaries and every piece gets that subtask's fixed instruction (real_split.instruction, the same strings the harness
sends). Pieces shorter than 1 s are dropped. Writes out/dagger_t3_908a.json and out/chk_dagger_t3_908a_N.jpg
(per episode: policy before the takeover, START, each boundary inside the human part, END)."""
import json, sys
from pathlib import Path
import av, numpy as np, pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import dagger_seg as DS, real_split as R
V1 = Path.home() / 'Downloads/b60f1e2e_908aa6e5/v1'; CAM = 'observation.images.camera_top'; FPS = 30
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


out = {}; cells = []; F = ImageFont.load_default(size=14)
for e in range(466, 475):
    m = np.where(ep == e)[0]; m = m[np.argsort(fi[m])]; T = len(m)
    segs, rm, gp = DS.segments(C[m], A[m], S[m])
    try:
        sp = R.segment(G[m], E[m]); bounds = [0, sp['b1'], sp['b2'], sp['b3'], T]; pan = sp['pan_arm']; err = None
    except ValueError as ex:
        sp, bounds, pan, err = None, None, None, str(ex)
    print(f'ep{e} T{T} {lab[e][1]} human {segs} | split {bounds} pan {pan} {err or ""}')
    pieces = []
    if segs and bounds:
        for st, en in segs:
            for k, sub in enumerate(R.SUBTASK_IDS):
                a, b = max(st, bounds[k]), min(en, bounds[k + 1])
                if b - a >= FPS:
                    pieces.append({'ep': e, 'start': int(a), 'end': int(b), 'subtask': sub, 'pan_arm': pan, 'secs': round((b - a) / FPS, 1),
                                   'task': R.instruction(sub, pan), 'result': lab[e][1]})
    for p in pieces:
        out[f"{e}_{p['subtask']}_{p['start']}"] = p
        print(f"   {p['subtask']:5s} {p['start']}-{p['end']} {p['secs']}s")
    # check strip: policy before, START, every boundary inside the human part (b-1 | b), END
    idx = []
    for st, en in (segs or []):
        idx += [max(st - 60, 0), st] + [x for b in (bounds or [])[1:4] if st < b < en for x in (b - 1, b)] + [en - 1]
    idx = idx[:10] or [0, T // 2, T - 1]
    fr = frames(e, idx)
    cell = Image.new('RGB', (10 * 192, 175), 'white')
    for k, i in enumerate(idx):
        im = fr[i].resize((192, 144)); sub = None
        if bounds:
            sub = next(R.SUBTASK_IDS[j] for j in range(4) if bounds[j] <= i < bounds[j + 1]) if i < T else 'home'
        cell.paste(im, (k * 192, 30)); ImageDraw.Draw(cell).text((k * 192 + 3, 16), f'f{i} {sub or ""}', font=F, fill=(160, 0, 0))
    ImageDraw.Draw(cell).text((4, 1), f"ep{e} ({lab[e][1]}) pan {pan} split {bounds} | human {segs} | pieces " +
                              ', '.join(f"{p['subtask']} {p['start']}-{p['end']}" for p in pieces), font=F, fill='black')
    cells.append(cell)
for k in range(0, len(cells), 5):
    cc = cells[k:k + 5]; sh = Image.new('RGB', (1920, 175 * len(cc)), 'white')
    for i, c_ in enumerate(cc): sh.paste(c_, (0, 175 * i))
    sh.save(f'out/chk_dagger_t3_908a_{k // 5 + 1}.jpg', quality=85)
json.dump(out, open('out/dagger_t3_908a.json', 'w'), indent=1, ensure_ascii=False)
print(len(out), 'pieces', {s: sum(1 for p in out.values() if p['subtask'] == s) for s in R.SUBTASK_IDS})
