"""Where do placed objects end up inside the organizer level box? For own task-2 demos with a recorded f0 destination box
(out/own*_cues.json), find the target object in the last frame (YOLO v13, target class, >= 0.25, closest to the dest box)
and report its position inside the dest box (0 = box top, 1 = box bottom). Sheet: last frame + f0 dest box (orange) +
placed object (white) + v13 level boxes on the last frame (thin)."""
import json, glob, sys, statistics as st
from pathlib import Path
import av, numpy as np, pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path.home() / 'lingbot_v7work')); import yolocue_bridge as YC
det = YC.Detector(YC.ROOT / 'weights/v13.torchscript', json.loads((YC.ROOT / 'weights/v13.names.json').read_text())['names'], conf=0.2)
V = Path.home() / 'Downloads/b60f1e2e_77235417/v1'; C = 'observation.images.camera_top'
meta = {m['episode_index']: m for m in pq.read_table(sorted((V / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()}
def last(e):
    m = meta[e]; rel = Path('videos') / C / f"chunk-{m[f'videos/{C}/chunk_index']:03d}" / f"file-{m[f'videos/{C}/file_index']:03d}.mp4"
    p = next(d_ / rel for d_ in [Path.home() / f'Downloads/{x}/v1' for x in ('b60f1e2e_77235417', 'b60f1e2e_3541a365', 'b60f1e2e_0d3bf9f6')] if (d_ / rel).exists()); t1 = m[f'videos/{C}/to_timestamp'] - 0.12
    with av.open(str(p)) as ic:
        s = ic.streams.video[0]; ic.seek(int(max(t1 - 1, 0) / s.time_base), stream=s, backward=True); b = None
        for fr in ic.decode(s):
            if float(fr.pts * s.time_base) > t1: break
            b = fr
        return b.to_ndarray(format='rgb24')
rows, cells = [], []; F = ImageFont.load_default(size=13)
for f in sorted(glob.glob(str(Path.home() / 'lingbot_v7work/out/own*_cues.json'))):
    E = json.load(open(f)).get('episodes', {})
    for k, v in E.items():
        if not str(v.get('dest', '')).startswith('organizer'): continue
        e = int(k); rgb = last(e); d = det(rgb); db = v['dest_box']; tcls = v['target']
        c = [x for x in d if x[0] == tcls]
        if not c: rows.append((e, v['dest'], None)); continue
        cx = lambda x: ((x[1] + x[3]) / 2, (x[2] + x[4]) / 2)
        dist = lambda x: max(0, db[0] - cx(x)[0], cx(x)[0] - db[2]) + max(0, db[1] - cx(x)[1], cx(x)[1] - db[3])
        o = min(c, key=dist); h = db[3] - db[1]
        pos = {'center': (cx(o)[1] - db[1]) / h, 'bottom': (o[4] - db[1]) / h, 'top': (o[2] - db[1]) / h, 'conf': o[5], 'outside_px': dist(o)}
        rows.append((e, v['dest'], pos))
        im = Image.fromarray(rgb); g = ImageDraw.Draw(im)
        for x in d:
            if x[0].startswith('organizer_') and x[5] >= 0.5: g.rectangle(x[1:5], outline=(120, 200, 255), width=1); g.text((x[1] + 2, x[2] + 1), x[0][10:], font=F, fill=(120, 200, 255))
        g.rectangle(db, outline=(255, 150, 0), width=3); g.rectangle(o[1:5], outline=(255, 255, 255), width=2)
        g.text((4, 4), f"ep{e} {tcls}->{v['dest'][10:]}  center {pos['center']:.2f} bottom {pos['bottom']:.2f} (0=top of orange box)", font=F, fill=(255, 255, 0)); cells.append(im)
for lv in ('organizer_top', 'organizer_middle', 'organizer_bottom'):
    P = [r[2] for r in rows if r[1] == lv and r[2]]
    if P: print(f"{lv}: n={len(P)}  object center at {st.median(p['center'] for p in P):.2f} (range {min(p['center'] for p in P):.2f}-{max(p['center'] for p in P):.2f}), object bottom at {st.median(p['bottom'] for p in P):.2f}, object top at {st.median(p['top'] for p in P):.2f}")
print('not found:', [r[0] for r in rows if not r[2]])
for k in range(0, len(cells), 6):
    cc = cells[k:k + 6]; sh = Image.new('RGB', (1920, 960), 'white')
    for i, c in enumerate(cc): sh.paste(c, ((i % 3) * 640, (i // 3) * 480))
    sh.save(Path.home() / f'lingbot_v7work/yolo13/levelpos_{k // 6 + 1}.jpg', quality=85)
json.dump([(r[0], r[1], r[2]) for r in rows], open(Path.home() / 'lingbot_v7work/yolo13/levelpos.json', 'w'), indent=1)
