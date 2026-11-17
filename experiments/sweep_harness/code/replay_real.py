"""Replay a scheduler on real-robot session recordings with the real LLM (DeepSeek-V4.1-Flash).

The recordings are eps 172-179 of one of our own platform datasets: whole 5080 sessions of task 3 (03:23-03:58 on 11/15) run by the
VLA under harness.Proprio, matched to their robotharness logs by gripper traces (local alignment file, not in the repo; error
< 0.01). The robot's motion is fixed (it does not react to the replayed scheduler), so this checks the scheduler's
judgements on real images and readings: when it would switch, and whether the scene supports the switch.
Cadence like the 5080 bridge: one on_chunk every CHUNK_S of session time; LLM replies are applied after their real
latency, one query in flight at a time.
  replay_real.py ALIGN_JSON RECORDING_V1 OUT_JSON [scheduler=proprio2] [episodes=172,...]"""
import json, sys, threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import av, numpy as np, pyarrow.parquet as pq
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.append(str(HERE.parent / 'deploy_5080' / 'deps'))
import harness as H

ALIGN, V, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
KIND = sys.argv[4] if len(sys.argv) > 4 else 'proprio2'
A = {int(k): v for k, v in json.load(open(ALIGN)).items() if v['err'] < 0.05}
for v in A.values():
    v['log'] = str((ALIGN.parent.parent / v['log']) if not Path(v['log']).is_absolute() else Path(v['log']))
if len(sys.argv) > 5: A = {e: A[e] for e in map(int, sys.argv[5].split(','))}
CHUNK_S = 0.8
CAMS = {'top': 'observation.images.camera_top', 'left': 'observation.images.camera_wrist_left', 'right': 'observation.images.camera_wrist_right'}
tb = pq.read_table(sorted((V / 'data').glob('*/*.parquet'))[0], columns=['episode_index', 'frame_index', 'observation.state.effector.position', 'observation.state.arm.position'])
EP, FR = tb.column('episode_index').to_numpy(), tb.column('frame_index').to_numpy()
GR, Q = np.array(tb.column('observation.state.effector.position').to_pylist()), np.array(tb.column('observation.state.arm.position').to_pylist())
META = {m['episode_index']: m for m in pq.read_table(sorted((V / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()}


def decode(e, cam, times):
    m = META[e]; p = V / 'videos' / cam / f"chunk-{m[f'videos/{cam}/chunk_index']:03d}" / f"file-{m[f'videos/{cam}/file_index']:03d}.mp4"
    t0 = m[f'videos/{cam}/from_timestamp']; out, k = {}, 0
    with av.open(str(p)) as c:
        s = c.streams.video[0]; c.seek(int(max(t0 + times[0] - 1, 0) / s.time_base), stream=s, backward=True)
        for fr in c.decode(s):
            while k < len(times) and float(fr.pts * s.time_base) + 1e-3 >= t0 + times[k]:
                out[k] = fr.to_ndarray(format='rgb24'); k += 1
            if k == len(times): break
    return [out[i] for i in range(len(times))]


class Clk:
    def __init__(self): self.now = 0.0
    def block(self, s): self.now += s


class Sync:
    """Executor stand-in: runs the LLM call at submit time; the scheduler still applies it after its latency."""
    def submit(self, fn, *a):
        f = Future(); f.set_result(fn(*a)); return f
    def shutdown(self, **k): pass


def run(e):
    a = A[e]; off = a['offset_s']; m = np.where(EP == e)[0]; m = m[np.argsort(FR[m])]; T = len(m) / 30
    ts = [i * CHUNK_S for i in range(int((T - off - 0.5) / CHUNK_S))]
    rec_t = [t + off for t in ts]; rec_t = [max(0.0, x) for x in rec_t]
    imgs = {k: decode(e, cam, rec_t) for k, cam in CAMS.items()}
    sch = H.make_scheduler(KIND); sch.pool = Sync(); clk = Clk()
    def obs(i):
        j = min(int(rec_t[i] * 30), len(m) - 1)
        return {'images': {k: imgs[k][i] for k in CAMS}, 'grippers': tuple(GR[m[j]]), 'arm': Q[m[j]]}
    sch.start(obs(0), clk)
    for i in range(1, len(ts)):
        clk.now = max(clk.now, ts[i])
        if clk.now >= ts[-1] or sch.done: break
        sch.on_chunk(obs(i), clk)
    sch.close(); s = sch.summary()
    trans = [(ev['t'], ev.get('subtask'), ev.get('why')) for ev in s['events'] if 'set' in ev]
    old = [(x['t'], x['subtask'], x.get('why')) for x in map(json.loads, open(a['log'])) if x.get('event') == 'prompt']
    print(f'ep{e} ({Path(a["log"]).name[22:28]}) {KIND}:', [(round(t, 1), st, (w or '')[:60]) for t, st, w in trans], '| old:', [(round(t, 1), st) for t, st, w in old], flush=True)
    return {'ep': e, 'log': a['log'], 'offset_s': off, 'transitions': trans, 'old_transitions': old, 'observations': s.get('observations'),
            'n_calls': s.get('n_calls'), 'n_invalid': s.get('n_invalid'), 'n_backs': s.get('n_backs'), 'stopped': sch.done}


with ThreadPoolExecutor(max_workers=8) as ex:
    res = list(ex.map(run, sorted(A)))
OUT.write_text(json.dumps(res, indent=1, default=str))
