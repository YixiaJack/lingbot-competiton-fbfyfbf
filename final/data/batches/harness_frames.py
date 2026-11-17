import json, glob, sys
from pathlib import Path
import av, numpy as np, pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
V1 = Path.home() / 'Downloads/105a2d09_41c236b9/v1'; CAM = 'observation.images.camera_top'
meta = {m['episode_index']: m for m in pq.read_table(sorted((V1 / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()}
A = {int(k): v for k, v in json.load(open('harness_logs/align_105.json')).items() if v['err'] < 0.05}
F = ImageFont.load_default(size=14)
def frames(e, ts):
    m = meta[e]; p = V1 / 'videos' / CAM / f"chunk-{m[f'videos/{CAM}/chunk_index']:03d}" / f"file-{m[f'videos/{CAM}/file_index']:03d}.mp4"; t0 = m[f'videos/{CAM}/from_timestamp']
    want = sorted(set(round(t, 2) for t in ts)); got = {}; k = 0
    with av.open(str(p)) as ic:
        s = ic.streams.video[0]; ic.seek(int(max(t0 + want[0] - 1, 0) / s.time_base), stream=s, backward=True)
        for fr in ic.decode(s):
            while k < len(want) and float(fr.pts * s.time_base) + 1e-3 >= t0 + want[k]: got[want[k]] = fr.to_image(); k += 1
            if k == len(want): break
    return got
mode = sys.argv[1]
rows = []
for e, a in sorted(A.items()):
    L = [json.loads(l) for l in open(a['log']) if l.strip()]; S = next(x for x in L if x.get('event') == 'summary'); ob = S['observations']
    tr = [(x['t'], x['subtask']) for x in L if x.get('event') == 'prompt']
    td = next((t for t, s in tr if s == 'dump'), None); th = next((t for t, s in tr if s == 'home'), None); tsw = next((t for t, s in tr if s == 'sweep'), None)
    off = a['offset_s']; T = meta[e]['length'] / 30
    if mode == 'dump': hts = [td - 3, td, td + 3, td + 6, td + 9, th - 2, th + 2, th + 6] if td and th else []
    elif mode == 'sweep': hts = [tsw - 2, tsw + 4, tsw + 10, (tsw + td) / 2, td - 6, td - 3, td - 1, td + 1] if tsw and td else []
    hts = [h for h in hts if 0 <= h + off < T - 0.2]
    if not hts: continue
    got = frames(e, [h + off for h in hts])
    row = Image.new('RGB', (len(hts) * 300, 262), 'white')
    for i, h in enumerate(hts):
        im = got[round(h + off, 2)].resize((300, 225)); sub = [s for t, s in tr if t <= h][-1]
        o = min(ob, key=lambda o: abs(o['t'] - h)); g = ImageDraw.Draw(im)
        g.rectangle([0, 0, 300, 17], fill=(0, 0, 0)); g.text((3, 2), f"t{h:.0f} {sub} | in/tab/hid {o['balls_in_dustpan']}/{o['balls_on_table']}/{o['balls_hidden']}", font=F, fill=(255, 255, 0))
        row.paste(im, (i * 300, 37))
    ImageDraw.Draw(row).text((3, 3), f"ep{e} = log {Path(a['log']).name[22:28]} | sweep@{tsw} dump@{td} home@{th}", font=F, fill='black')
    rows.append(row)
for k in range(0, len(rows), 4):
    rr = rows[k:k + 4]; W = max(r.width for r in rr); sh = Image.new('RGB', (W, 262 * len(rr)), 'white')
    for i, r in enumerate(rr): sh.paste(r, (0, 262 * i))
    sh.save(f'harness_logs/{mode}_{k // 4 + 1}.jpg', quality=82)
print(len(rows))
