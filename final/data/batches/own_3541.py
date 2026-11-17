"""Normal teleop demos of b60f1e2e 3541a365 (10:17): eps 374-388 (task 2: 374-384, task 1: 385-388) -> official-distribution prompt rewrite, v9 box cues
with the own-data grasp checks, plain copies. No DAgger recordings in this batch (dagger_seg found none)."""
import random, re, collections
import json, subprocess, sys
from pathlib import Path
import av, numpy as np, pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '.'); sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import yolocue_bridge as YC, derive_dataset as DD, real_split as R
V1 = Path.home() / 'Downloads/b60f1e2e_3541a365/v1'; O = Path('out/ds_own3541'); O.mkdir(parents=True, exist_ok=True); CAM = 'observation.images.camera_top'
fps = json.loads((V1 / 'meta/info.json').read_text())['fps']
meta = {m['episode_index']: m for m in pq.read_table(sorted((V1 / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()}
lab = {}
for l in open(V1 / 'meta/labels.jsonl'):
    r = json.loads(l); L = (r.get('labels') or [{}])[0]; lab[r['episode_index']] = (L.get('task_instruction') or '', L.get('result'))
NEW = list(range(374, 389))
clean = lambda s: re.sub(r'\b(onion|mangosteen) fruit\b', r'\1', re.sub(r'\bstar fruit\b', 'starfruit', s, flags=re.I), flags=re.I)
T12 = [e for e in NEW if YC.parse(clean(lab[e][0]))]; T3 = []
print('task-1/2', T12, '| not parsed', [e for e in NEW if e not in T12], '| results', collections.Counter(lab[e][1] for e in NEW))
pol = json.load(open(Path.home() / '.cache/field_dev/watch/repo/prompt_policy.json'))['tasks']
OWN = {'starfruit': 'starfruit', 'green_chili': 'green chili pepper', 'glitter_compact': 'black round glitter compact', 'brush_small': 'small black and gold makeup brush'}
def allocate(n, weights, seed):
    tot = sum(weights); raw = [n * w / tot for w in weights]; cnt = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - cnt[i], reverse=True)[:n - sum(cnt)]: cnt[i] += 1
    seq = [i for i, c in enumerate(cnt) for _ in range(c)]; random.Random(seed).shuffle(seq); return seq
by = collections.defaultdict(list)
for e in T12: pr = YC.parse(clean(lab[e][0])); by[pr[0]].append((e, pr))
instr = {}
for t, lst in by.items():
    TT = pol[str(t)]; seq = allocate(len(lst), [x['weight'] for x in TT['templates']], f'own3541:{t}')
    for (e, pr), i in zip(lst, seq):
        obj = TT['object_phrases'].get(pr[1]) or OWN[pr[1]]
        txt = TT['templates'][i]['text'].format(object=obj, destination=pr[2].split('_', 1)[1]); assert YC.parse(txt) == pr, (e, txt); instr[e] = txt
for e, s_ in instr.items(): print(f'  instr {e}: {lab[e][0]!r} -> {s_!r}')
json.dump({str(k): v for k, v in instr.items()}, open('instr_3541.json', 'w'), ensure_ascii=False, indent=1)
Path('instr_3541_dbd.json').write_text(json.dumps({'episodes': {str(k): {'instruction': v} for k, v in instr.items()}}, ensure_ascii=False))
t = pq.read_table(sorted((V1 / 'data').glob('*/*.parquet'))[0], columns=['episode_index', 'frame_index', 'action.effector.position', 'observation.state.effector.position', 'observation.state.ee.position'])
ep, fi = t.column('episode_index').to_numpy(), t.column('frame_index').to_numpy()
C = np.array(t.column('action.effector.position').to_pylist(), float); G = np.array(t.column('observation.state.effector.position').to_pylist(), float); E = np.array(t.column('observation.state.ee.position').to_pylist(), float)
def rows(e):
    m = np.where(ep == e)[0]; return m[np.argsort(fi[m])]
def frames(e, idx):
    m = meta[e]; p = V1 / 'videos' / CAM / f"chunk-{m[f'videos/{CAM}/chunk_index']:03d}" / f"file-{m[f'videos/{CAM}/file_index']:03d}.mp4"; t0 = m[f'videos/{CAM}/from_timestamp']
    want = sorted(set(idx)); ts = [t0 + i / fps for i in want]; got = {}; k = 0
    with av.open(str(p)) as ic:
        s = ic.streams.video[0]; ic.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s, backward=True)
        for fr in ic.decode(s):
            while k < len(want) and float(fr.pts * s.time_base) + 1e-3 >= ts[k]: got[want[k]] = fr.to_ndarray(format='rgb24'); k += 1
            if k == len(want): break
    return got
def blocked(c, g):
    best = None
    for x in (0, 1):
        if c[:, x].max() < 0.25: continue
        j0 = int(np.argmax(c[:, x] > 0.25)); run = 0
        for j in range(j0, len(c)):
            run = run + 1 if g[j, x] - c[j, x] >= 0.08 else 0
            if run == 10: best = (j - 9, x) if best is None or j - 9 < best[0] else best; break
    return best
def cmd_grasp(c):
    best = None
    for x in (0, 1):
        op = False
        for j, v in enumerate(c[:, x]):
            if not op and v > YC.CMD_OPEN: op = True
            elif op and v < YC.CMD_CLOSED: best = (j, x) if best is None or j < best[0] else best; break
    return best
F = ImageFont.load_default(size=14); det = YC.detector(YC.load_config()); print('detector', YC.load_config()['weights'])
cues = {}; cells = []
for e in T12:
    m = rows(e); c, g, ee = C[m], G[m], E[m]; Tn = len(m); task, tcls, dcls = YC.parse(instr[e])
    cands = [x for x in (cmd_grasp(c), blocked(c, g)) if x]; gf, arm = min(cands) if cands else (None, None)
    arms = [x for x in (0, 1) if (c[:, x] > 0.25).any()]
    hold = float(np.median(g[gf + 15:gf + 30, arm]) - np.median(g[:15, arm])) if gf is not None else None
    lift = float(ee[gf:gf + 60, 7 * arm + 2].max() - ee[gf, 7 * arm + 2]) if gf is not None else None
    gfx = gf if gf is not None else Tn // 2
    fr = frames(e, [0, gfx, min(gfx + 30, Tn - 1), Tn - 3]); a = fr[0]
    tb, db, why = YC.select(det(a), task, tcls, dcls, rgb=a)
    if gf is None: tb, db, why = None, None, 'no grasp frame (gripper command never closes after opening)'
    print(f'  ep{e} cue {why} gf {gf} (cmd {cmd_grasp(c)}, blocked {blocked(c, g)}) arms {arms} hold {hold} lift {lift}')
    if tb: cues[str(e)] = {'id': f'ep{e:04d}', 'instruction': instr[e], 'target': tcls, 'dest': dcls, 'target_box': tb, 'dest_box': db, 'grasp_frame': int(gf), 'T': Tn}
    pan = [Image.fromarray(YC.draw(a, tb, db, False)) if tb else Image.fromarray(a)]
    for k in (gfx, min(gfx + 30, Tn - 1)):
        im = Image.fromarray(fr[k]); d2 = ImageDraw.Draw(im)
        if tb: d2.rectangle(tb, outline=(255, 255, 255), width=2)
        pan.append(im)
    pan.append(Image.fromarray(fr[Tn - 3]))
    cell = Image.new('RGB', (1920, 380), 'white')
    for k, im in enumerate(pan): cell.paste(im.resize((480, 360)), (k * 480, 20))
    ImageDraw.Draw(cell).text((4, 3), f'ep{e} ({lab[e][1]}) {tcls}->{dcls} | {why} | gf {gf} arm {arm} arms {arms} hold {hold if hold is None else round(hold, 2)} lift {lift if lift is None else round(lift, 2)} | {instr[e]} | f0, gf, gf+30, last', font=F, fill='black'); cells.append(cell)
for k in range(0, len(cells), 5):
    cc = cells[k:k + 5]; sh = Image.new('RGB', (1920, 380 * len(cc)), 'white')
    for i, c_ in enumerate(cc): sh.paste(c_, (0, 380 * i))
    sh.save(f'out/chk3541_t12_{k // 5 + 1}.jpg', quality=82)
json.dump({'summary': {'yolo': YC.load_config()['weights'], 'checked_by_eye': 'pending', 'mode': 'boxgrasp', 'line_px': 5}, 'episodes': cues}, open('out/own3541_cues.json', 'w'), indent=1, ensure_ascii=False)
segs = {}; cells = []
for e in T3:
    m = rows(e); Tn = len(m)
    try: s = R.segment(G[m], E[m])
    except ValueError as ex: print('  ep', e, 'split failed:', ex); continue
    b = [0, s['b1'], s['b2'], s['b3'], s['T']]; segs[e] = (b, s['pan_arm']); print(f'  ep{e} pan_arm {s["pan_arm"]} bounds {b} secs {[round((b[i+1]-b[i])/fps,1) for i in range(4)]}')
    idx = [0, b[1] - 1, b[1], b[2] - 1, b[2], b[3] - 1, b[3], Tn - 3]; fr = frames(e, idx)
    cell = Image.new('RGB', (8 * 240, 200), 'white')
    for k, i in enumerate(idx):
        cell.paste(Image.fromarray(fr[i]).resize((240, 180)), (k * 240, 20))
    ImageDraw.Draw(cell).text((4, 3), f'ep{e} ({lab[e][1]}) pan {s["pan_arm"]} | f0 | grasp end/sweep start {b[1]} | sweep end/dump start {b[2]} | dump end/home start {b[3]} | last', font=F, fill='black'); cells.append(cell)
if cells:
    sh = Image.new('RGB', (1920, 200 * len(cells)), 'white')
    for i, c_ in enumerate(cells): sh.paste(c_, (0, 200 * i))
    sh.save('out/chk3541_t3.jpg', quality=85)
json.dump({str(k): {'bounds': v[0], 'pan_arm': v[1]} for k, v in segs.items()}, open('out/own3541_t3split.json', 'w'), indent=1)
