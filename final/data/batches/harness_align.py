"""Match 105a2d09 41c236b9 eps 172-181 (whole-session recordings) to robotharness logs by gripper traces, and save the
mapping (episode, log, time offset). Harness gripper readings come from the 'user' text of each LLM call
("Time X s ... Grippers: left a, right b."), already x10 = dataset units."""
import json, glob, re, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
V1 = Path.home() / 'Downloads/105a2d09_41c236b9/v1'
t = pq.read_table(sorted((V1 / 'data').glob('*/*.parquet'))[0], columns=['episode_index', 'frame_index', 'observation.state.effector.position'])
ep, fi = t.column('episode_index').to_numpy(), t.column('frame_index').to_numpy(); G = np.array(t.column('observation.state.effector.position').to_pylist())
logs = {}
for f in sorted(glob.glob('harness_logs/robotharness_20261115_0[3-4]*.jsonl')):
    L = [json.loads(l) for l in open(f) if l.strip()]; pts = []
    for x in L:
        if x.get('event') != 'llm': continue
        u = x.get('user'); s = u if isinstance(u, str) else json.dumps(u)
        m = re.search(r'Time ([\d.]+) s.*?Grippers: left (-?\d+\.\d+), right (-?\d+\.\d+)', s)
        if m: pts.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
    if len(pts) > 10: logs[f] = np.array(pts)
out = {}
for e in range(172, 182):
    m = np.where(ep == e)[0]; m = m[np.argsort(fi[m])]; g = G[m]; T = len(m) / 30
    best = None
    for f, P in logs.items():
        for off in np.arange(-15, 15.01, 0.25):   # recording time = harness time + off
            idx = np.round((P[:, 0] + off) * 30).astype(int); ok = (idx >= 0) & (idx < len(m))
            if ok.sum() < 0.7 * len(P): continue
            err = np.abs(g[idx[ok]] - P[ok, 1:]).mean()
            if best is None or err < best[0]: best = (err, f, off, int(ok.sum()), len(P))
    out[e] = {'log': best[1], 'offset_s': float(best[2]), 'err': round(float(best[0]), 4), 'n': best[3], 'n_log': best[4], 'rec_s': round(T, 1)}
    print(e, f'{T:.0f}s', Path(best[1]).name[13:28], 'offset', best[2], 'err', round(best[0], 4), best[3], '/', best[4])
json.dump(out, open('harness_logs/align_105.json', 'w'), indent=1)
