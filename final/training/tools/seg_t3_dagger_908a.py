"""Task-3 DAgger episodes 466-474 of b60f snapshot 908aa6e5: subtask boundaries (real_split.segment on the whole
episode) intersected with the human takeover ranges from dagger_seg; writes labels JSON for derive_dataset --mode
t3_subtasks and contact sheets (camera_top) at the cut points.
usage: python seg908a.py SRC_V1 OUT_DIR [--sheets]"""
import sys, json, glob, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/lingbot-final-notes/experiments/sweep_harness/code'))
import real_split as rs

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
H = {466: [(2195, 3760)], 467: [(510, 2346), (2396, 2546)], 468: [(1331, 2352)], 469: [(515, 1568), (1604, 1809)],
     470: [(375, 1455)], 471: [(413, 1649)], 472: [(315, 2330)], 473: [(440, 1459)], 474: [(291, 1560)]}
MIN_PIECE = 30
d = pd.read_parquet(glob.glob(f'{src}/data/*/*.parquet')[0], columns=['episode_index', 'frame_index', 'observation.state.effector.position', 'observation.state.ee.position'])
d = d[d.episode_index.isin(list(H))]
eps, rows = [], []
for ep, segs in H.items():
    x = d[d.episode_index == ep].sort_values('frame_index')
    gr = np.stack(x['observation.state.effector.position'].values); ee = np.stack(x['observation.state.ee.position'].values)
    try:
        s = rs.segment(gr, ee)
    except ValueError as ex:
        # 469: the brush (right) gripper never opens past 0.38 at the end; take the release from a lower threshold on it
        s = None
        for thr in (0.30, 0.25):
            g2 = gr.copy()
            r_open = [(a, b) for a, b in rs.runs(gr[:, 1] > thr) if b - a >= 5]
            if len(r_open) >= 2:
                g2[r_open[-1][0]:r_open[-1][1], 1] = 0.5
                try:
                    s = rs.segment(g2, ee); s['note'] = f'release on brush arm from thr {thr}'; break
                except ValueError:
                    pass
        if s is None:
            print(ep, 'SKIP', ex); continue
    lab = rs.frame_labels(s)
    pieces = []
    for a, b in segs:
        cur, st = lab[a], a
        for i in range(a + 1, b + 1):
            if i == b or lab[i] != cur:
                if i - st >= MIN_PIECE:
                    pieces.append({'start': st, 'end': i, 'subtask': cur, 'instruction': rs.instruction(cur, s['pan_arm'])})
                else:
                    print(f'  {ep}: drop short {cur} piece [{st},{i}) {i - st} frames')
                if i < b:
                    cur, st = lab[i], i
    eps.append({'id': f'ep{ep:04d}', 'episode_index': ep, 'pan_arm': s['pan_arm'], 'segments': pieces,
                'b': [s['b1'], s['b2'], s['b3'], s['release']], 'human': segs, 'note': s.get('note', '')})
    print(ep, s['pan_arm'], 'b1-3/rel', s['b1'], s['b2'], s['b3'], s['release'], 'hold', s['hold_pan'], s['hold_brush'], s.get('note', ''),
          '|', ' '.join(f"{p['subtask']}[{p['start']},{p['end']})" for p in pieces))
json.dump({'episodes': eps}, open(f'{out}/t3_dagger_908a_labels.json', 'w'), indent=1)
tot = {k: sum(p['end'] - p['start'] for e in eps for p in e['segments'] if p['subtask'] == k) for k in rs.SUBTASK_IDS}
print('frames per subtask', tot, 'pieces', sum(len(e['segments']) for e in eps))

if '--sheets' in sys.argv:
    import av
    from PIL import Image, ImageDraw
    info = json.load(open(f'{src}/meta/info.json'))
    em = pd.read_parquet(glob.glob(f'{src}/meta/episodes/*/*.parquet')[0])
    k = 'observation.images.camera_top'
    for e in eps:
        r = em[em.episode_index == e['episode_index']].iloc[0]
        vf = f"{src}/" + info['video_path'].format(video_key=k, chunk_index=int(r[f'videos/{k}/chunk_index']), file_index=int(r[f'videos/{k}/file_index']))
        t0 = float(r[f'videos/{k}/from_timestamp'])
        marks = [(e['human'][0][0], 'human start')]
        for p in e['segments']:
            marks += [(p['start'] + 5, f"{p['subtask']} start"), (p['end'] - 5, f"{p['subtask']} end")]
        marks.append((e['human'][-1][1] - 1, 'human end'))
        want = sorted({(f, l) for f, l in marks})
        frames = {}
        with av.open(vf) as c:
            st = c.streams.video[0]
            for f, l in want:
                ts = t0 + f / info['fps']
                c.seek(int(max(0, ts - 1.0) / st.time_base), stream=st, backward=True)
                for fr in c.decode(st):
                    if fr.time is not None and fr.time >= ts - 0.5 / info['fps']:
                        frames[(f, l)] = fr.to_image().resize((320, 240)); break
        W = 4; n = len(want); rows_ = (n + W - 1) // W
        sheet = Image.new('RGB', (320 * W, 262 * rows_), 'white'); dr = ImageDraw.Draw(sheet)
        for i, (f, l) in enumerate(want):
            x0, y0 = 320 * (i % W), 262 * (i // W)
            if (f, l) in frames: sheet.paste(frames[(f, l)], (x0, y0 + 22))
            dr.text((x0 + 4, y0 + 4), f"ep{e['episode_index']} f{f} {l}", fill='black')
        sheet.save(f"{out}/ep{e['episode_index']}.jpg", quality=80)
    print('sheets in', out)
