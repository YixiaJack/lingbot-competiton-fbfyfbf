"""Extra box cues, second pass: episodes whose gripper command never closes below 0.2 while holding (partial close),
or whose YOLO target box was wrong / missing. Grasp frame = first frame after the gripper opened where the measured
gripper stays >= 0.08 above the command for 10 frames (the object blocks the fingers). Cue from extra_props, YOLO
select, or the by-eye-checked yolo8 labels (yolo8/ds/final_labels.json). Renders frame 0 + cue, grasp frame,
grasp+30 and the last frame for a by-eye check -> out/extra2_*.jpg, out/extra2_props.json."""
import json, sys
from pathlib import Path
import av, numpy as np, pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '.')
import yolocue_bridge as YC, v7_boxes as VB

V1 = Path.home() / 'Downloads/b60f1e2e_0d3bf9f6/v1'; CAM = VB.CAM
meta = {m['episode_index']: m for m in pq.read_table(sorted((V1 / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()}
Cd = {int(k): v for k, v in json.load(open('out/extra_cands.json')).items()}
P1 = {int(k): v for k, v in json.load(open('out/extra_props.json')).items()}
LAB = json.load(open('yolo8/ds/final_labels.json'))['labels']
det = YC.detector(YC.load_config())
EPS = [int(x) for x in sys.argv[1].split(',')]
FROM_LABELS = {225, 260, 261}

t = pq.read_table(sorted((V1 / 'data').glob('*/*.parquet'))[0], columns=['episode_index', 'frame_index', 'action.effector.position',
                  'observation.state.effector.position', 'observation.state.ee.position'])
ep, fi = t.column('episode_index').to_numpy(), t.column('frame_index').to_numpy()
C = np.array(t.column('action.effector.position').to_pylist(), float); G = np.array(t.column('observation.state.effector.position').to_pylist(), float)
E = np.array(t.column('observation.state.ee.position').to_pylist(), float)


def frames(m, idx):
    p = V1 / 'videos' / CAM / f"chunk-{m[f'videos/{CAM}/chunk_index']:03d}" / f"file-{m[f'videos/{CAM}/file_index']:03d}.mp4"; t0 = m[f'videos/{CAM}/from_timestamp']
    want = sorted(set(idx)); got = {}; ts = [t0 + i / 30 for i in want]
    with av.open(str(p)) as ic:
        s = ic.streams.video[0]; ic.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s, backward=True); k = 0
        for fr in ic.decode(s):
            while k < len(want) and float(fr.pts * s.time_base) + 1e-3 >= ts[k]: got[want[k]] = fr.to_ndarray(format='rgb24'); k += 1
            if k == len(want): break
    return got


def blocked_grasp(c, g):
    best = None
    for x in (0, 1):
        if c[:, x].max() < 0.25: continue
        j0 = int(np.argmax(c[:, x] > 0.25)); run = 0
        for j in range(j0, len(c)):
            run = run + 1 if g[j, x] - c[j, x] >= 0.08 else 0
            if run == 10:
                j = j - 9
                if best is None or j < best[0]: best = (j, x)
                break
    return best


F = ImageFont.load_default(size=13); cells = []; props = {}
for n, e in enumerate(EPS, 1):
    v = Cd[e]; task, tcls, dcls = v['parsed']
    m = np.where(ep == e)[0]; m = m[np.argsort(fi[m])]; c, g, ee = C[m], G[m], E[m]; T = len(m)
    bg = blocked_grasp(c, g)
    if bg is None: print(e, 'no blocked grasp'); continue
    gf, x = bg
    hold = float(np.median(g[gf + 15:gf + 30, x]) - np.median(g[:15, x])); lift = float(ee[gf:gf + 60, 7 * x + 2].max() - ee[gf, 7 * x + 2])
    fr = frames(meta[e], [0, gf, min(gf + 30, T - 1), T - 3]); a = fr[0]
    if e in FROM_LABELS:
        L = {k: b for k, b in LAB[str(e)]}; tb, db, how = L.get(tcls), L.get(dcls), 'yolo8 labels'
    elif e in P1:
        tb, db, how = P1[e]['target_box'], P1[e]['dest_box'], 'props: ' + P1[e]['how']
    else:
        tb, db, how = YC.select(det(a), task, tcls, dcls, rgb=a); how = 'yolo ' + how
    props[e] = {'id': f'ep{e:04d}', 'instruction': v['instruction'], 'target': tcls, 'dest': dcls, 'target_box': tb, 'dest_box': db,
                'grasp_frame': int(gf), 'T': T, 'how': how, 'arm': x, 'hold': round(hold, 3), 'lift': round(lift, 3), 'old_gf': v['grasp_frame']}
    p0 = Image.fromarray(YC.draw(a, tb, db, False)) if tb and db else Image.fromarray(a)
    pan = [p0]
    for k in (gf, min(gf + 30, T - 1)):
        im = Image.fromarray(fr[k]); d2 = ImageDraw.Draw(im)
        if tb: d2.rectangle(tb, outline=(255, 255, 255), width=2)
        pan.append(im)
    pan.append(Image.fromarray(fr[T - 3]))
    cell = Image.new('RGB', (1920, 380), (255, 255, 255))
    for k, im in enumerate(pan): cell.paste(im.resize((480, 360)), (k * 480, 20))
    ImageDraw.Draw(cell).text((4, 3), f"#{n} ep{e} {tcls}->{dcls} | {how} | arm{x} hold {hold:.2f} lift {lift:.2f} | gf {gf} (old {v['grasp_frame']}) | {v['instruction'][:60]} | panels: f0, gf, gf+30, last",
                              font=F, fill=(0, 0, 0) if tb and db else (200, 0, 0))
    cells.append(cell.resize((1600, 317)))
    print(e, json.dumps(props[e], ensure_ascii=False)[:260])
for k in range(0, len(cells), 5):
    cc = cells[k:k + 5]; sheet = Image.new('RGB', (1600, len(cc) * 317), (255, 255, 255))
    for i, cl in enumerate(cc): sheet.paste(cl, (0, i * 317))
    sheet.save(f'out/extra2_{k // 5 + 1}.jpg', quality=85)
json.dump({str(k): v for k, v in props.items()}, open('out/extra2_props.json', 'w'), indent=1, ensure_ascii=False)
