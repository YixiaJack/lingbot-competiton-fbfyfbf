"""Offline test of Proprio's ball counting with the real LLM (DeepSeek-V4.1-Flash, PROPRIO_SYSTEM) on official demos.
Closed-loop conversation per demo like the harness: first turn at frame 0, then one turn every STEP frames (three
cameras + grippers + running subtask from the labels), up to 1.5 s after the dustpan starts to lift (b2). The rule:
from SWEEP_MIN_S into the sweep, ALL_IN_N replies in a row with balls_in_dustpan >= balls_total and none on the table.
Every ball is in the pan by b2 (216/216 checked by eye), so firing well before b2 = premature.
  judge_balls.py DATASET_V1 OUT.json [n_demos] [step_frames]"""
import json, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import av, numpy as np, pyarrow.parquet as pq
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.append(str(HERE.parent / 'deploy_5080' / 'deps'))
import harness as H, real_split as RS
V, OUT = Path(sys.argv[1]), Path(sys.argv[2]); NDEMO = int(sys.argv[3]) if len(sys.argv) > 3 else 24
STEP = int(sys.argv[4]) if len(sys.argv) > 4 else 36
CAMS = ['observation.images.camera_top', 'observation.images.camera_wrist_left', 'observation.images.camera_wrist_right']
tb = pq.read_table(V / 'data/chunk-000/file-000.parquet', columns=['episode_index', 'frame_index', 'observation.state.effector.position', 'observation.state.ee.position'])
col = lambda k, n: tb.column(k).combine_chunks().flatten().to_numpy().reshape(-1, n)
EP, FR = tb.column('episode_index').to_numpy(), tb.column('frame_index').to_numpy()
GR, EE = col('observation.state.effector.position', 2), col('observation.state.ee.position', 14)
em = {r['episode_index']: r for r in pq.read_table(V / 'meta/episodes/chunk-000/file-000.parquet').to_pylist()}
lab = {json.loads(l)['episode_index']: json.loads(l)['labels'][0]['custom_fields'][0]['value'] for l in open(V / 'meta/labels.jsonl')}
info = json.load(open(V / 'meta/info.json'))

def frames(ep, cam, idxs):
    r = em[ep]; t0 = r[f'videos/{cam}/from_timestamp']
    path = V / info['video_path'].format(video_key=cam, chunk_index=r[f'videos/{cam}/chunk_index'], file_index=r[f'videos/{cam}/file_index'])
    out, want, k = {}, sorted(idxs), 0
    with av.open(str(path)) as c:
        s = c.streams.video[0]; c.seek(int(max(0, t0 - 1) / s.time_base), stream=s, backward=True)
        for f in c.decode(s):
            j = int(round((float(f.pts * s.time_base) - t0) * 30))
            while k < len(want) and want[k] < j: k += 1
            if k < len(want) and want[k] == j: out[j] = f.to_ndarray(format='rgb24'); k += 1
            if k >= len(want): break
    return out

def run(e):
    ix = np.where(EP == e)[0]; ix = ix[np.argsort(FR[ix])]
    s = RS.segment(GR[ix], EE[ix]); gr = GR[ix]
    idxs = list(range(0, min(s['b2'] + 45, s['T']), STEP))
    fr = {cam: frames(e, cam, idxs) for cam in CAMS}
    idxs = [j for j in idxs if all(j in fr[c] for c in CAMS)]
    chat = H.Chat(H.PROPRIO_SYSTEM); rows = []
    for n, j in enumerate(idxs):
        sub = 'grasp' if j < s['b1'] else 'sweep' if j < s['b2'] else 'dump'
        head = 'Start of the task. Time 0.0 s of 120 s.' if n == 0 else f'Time {j / 30:.1f} s of 120 s. Running now: subtask {sub}.'
        text = f'{head} Grippers: left {gr[j, 0]:.2f}, right {gr[j, 1]:.2f}.'
        content = [{'type': 'text', 'text': text}] + [H.jpeg_part(fr[c][j], size=(320, 240)) for c in CAMS]
        parsed, rec = chat.ask(content, text)
        rows.append({'frame': j, 'sub': sub, 'reply': parsed, 'error': rec['error'], 'latency': rec['latency_s']})
    return {'ep': e, 'id': lab[e], 'b1': s['b1'], 'b2': s['b2'], 'rows': rows}

eps = [e for e, i in lab.items() if 'T3_reserve' in i] + [e for e, i in lab.items() if i.startswith('T3_train')][::12]
eps = eps[:NDEMO]
with ThreadPoolExecutor(6) as ex:
    res = [r for r in ex.map(run, eps)]
OUT.write_text(json.dumps(res, default=str))
# rule
fire_err, prem, never, tot_ok = [], [], 0, 0
for r in res:
    first = r['rows'][0]['reply'] or {}
    n_total = None; n = 0; fire = None
    for x in r['rows']:
        d = x['reply'] or {}
        try: t, i_, o = int(d.get('balls_total')), int(d.get('balls_in_dustpan')), int(d.get('balls_on_table'))
        except (TypeError, ValueError): n = 0; continue
        if n_total is None: n_total = t
        elif x['sub'] == 'grasp' and t > n_total: n_total = t
        if x['sub'] in ('sweep', 'dump') and x['frame'] >= r['b1'] + 3 * 30:
            n = n + 1 if (i_ >= n_total and o == 0) else 0
            if n >= 2: fire = x['frame']; break
    if fire is None: never += 1; continue
    err = (fire - r['b2']) / 30; fire_err.append(err)
    if err < -2: prem.append((r['id'], round(err, 1)))
print(f'demos {len(res)} | fired {len(fire_err)} | never by b2+1.5 s {never}')
if fire_err: print('fire - b2 (s): p10 %.1f median %.1f p90 %.1f' % tuple(np.percentile(fire_err, [10, 50, 90])))
print('more than 2 s before b2:', prem)
