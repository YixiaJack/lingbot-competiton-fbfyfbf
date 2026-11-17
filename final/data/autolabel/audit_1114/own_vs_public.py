"""Does our own data look different from the official data in ways that could hurt training? Per-episode metrics on the
raw datasets, normalised the way training does (meanstd with the v2 norm stats; arm actions relative to the current
joint state over a 50-step chunk, gripper actions absolute):
  dur_s, idle_start_s, idle_end_s, longest_pause_s, speed (rad/s, moving frames), jerk, openings (gripper-command openings, both arms), track_err (|action - state| arm, rad), z_out (share of chunk actions with
  |z| > 3), z_max, state_z_out, brightness / saturation of camera_top (frame 0 and middle)
usage: own_vs_public.py NORM_JSON OUT_JSON public=DIR own=DIR[:EPS_JSON]"""
import json, sys, re
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import av

FPS, CH = 30, 50
N = json.loads(Path(sys.argv[1]).read_text())['norm_stats']
mu_a, sd_a = np.array(N['action.arm.position']['mean']), np.array(N['action.arm.position']['std'])
mu_g, sd_g = np.array(N['action.effector.position']['mean']), np.array(N['action.effector.position']['std'])
mu_s, sd_s = np.array(N['observation.state.arm.position']['mean']), np.array(N['observation.state.arm.position']['std'])


def task_of(ins, cid):
    if re.match(r'T[123]_', cid or ''): return int(cid[1])
    s = ins.lower()
    return 3 if ('dustpan' in s or 'paper ball' in s) else 1 if 'plate' in s else 2


def col(t, c):
    a = t.column(c).combine_chunks(); return a.flatten().to_numpy(zero_copy_only=False).reshape(len(a), -1)


def runs(m):
    m = np.r_[False, m, False].astype(int); d = np.diff(m); return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def img_stats(d, E, ep, fracs=(0.0, 0.5)):
    r = E[E.episode_index == ep].iloc[0]; k = 'observation.images.camera_top'
    path = d / f"videos/{k}/chunk-{int(r[f'videos/{k}/chunk_index']):03d}/file-{int(r[f'videos/{k}/file_index']):03d}.mp4"
    t0, t1 = float(r[f'videos/{k}/from_timestamp']), float(r[f'videos/{k}/to_timestamp']); out = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        for f in fracs:
            t = t0 + f * (t1 - t0 - 0.1); c.seek(int(max(t - 2, 0) / s.time_base), stream=s)
            for fr in c.decode(s):
                if fr.time is not None and fr.time >= t:
                    hsv = np.asarray(fr.to_image().convert('HSV').resize((160, 120)), dtype=float)
                    out.append((hsv[..., 2].mean() / 255, hsv[..., 1].mean() / 255)); break
    return out


def analyse(nm, d, keep=None):
    d = Path(d); lab = {}
    for line in open(d / 'meta/labels.jsonl'):
        r = json.loads(line); lb = r['labels'][0]; cf = {f['name']: f['value'] for f in (lb.get('custom_fields') or [])}
        lab[r['episode_index']] = (lb['task_instruction'], cf.get('id', ''), lb.get('result'))
    E = pd.concat(pd.read_parquet(f) for f in sorted((d / 'meta/episodes').glob('*/*.parquet')))
    t = pa.concat_tables([pq.read_table(f, columns=['episode_index', 'frame_index', 'observation.state.arm.position', 'action.arm.position', 'action.effector.position'])
                          for f in sorted((d / 'data').glob('*/*.parquet'))])
    ep = t.column('episode_index').to_numpy(); fi = t.column('frame_index').to_numpy()
    S, A, G = col(t, 'observation.state.arm.position'), col(t, 'action.arm.position'), col(t, 'action.effector.position')
    rows = []
    for e in sorted(lab):
        if keep is not None and e not in keep: continue
        m = np.nonzero(ep == e)[0]; m = m[np.argsort(fi[m])]; s, a, g = S[m], A[m], G[m]; T = len(m)
        ins, cid, res = lab[e]; task = task_of(ins, cid)
        v = np.linalg.norm(np.diff(a, axis=0), axis=1) * FPS                         # arm command speed, rad/s
        gv = np.abs(np.diff(g, axis=0)).max(1) * FPS
        moving = (v > 0.05) | (gv > 0.2)
        mv = np.nonzero(moving)[0]
        if len(mv) == 0: continue
        first, last = mv[0], mv[-1]
        pauses = [(b - a_) for a_, b in runs(~moving[first:last]) ]
        opens = [sum(1 for u, w in runs(g[:, k] > 0.5) if w - u >= 5) for k in (0, 1)]
        idx = np.arange(0, T - 1, 5); ks = np.array([0, 10, 20, 30, 40, 49])
        rel = np.stack([a[np.minimum(i + ks, T - 1)] - s[i] for i in idx])          # (n, 6, 12) chunk actions relative to state
        za = (rel - mu_a) / sd_a; zg = (g[np.minimum(idx[:, None] + ks, T - 1)] - mu_g) / sd_g
        zs = (s - mu_s) / sd_s
        im = img_stats(d, E, e)
        rows.append(dict(src=nm, ep=int(e), task=task, result=res, dur_s=T / FPS, idle_start_s=first / FPS, idle_end_s=(T - 1 - last) / FPS,
                         longest_pause_s=(max(pauses) if pauses else 0) / FPS, speed=float(np.median(v[moving[:len(v)]])) if moving.any() else 0.0,
                         jerk=float(np.abs(np.diff(a, n=2, axis=0)).mean() * FPS * FPS), openings=int(sum(opens)),
                         track_err=float(np.abs(a - s).mean()), z_out=float((np.abs(za) > 3).mean()), z_max=float(np.abs(za).max()),
                         zg_out=float((np.abs(zg) > 3).mean()), state_z_out=float((np.abs(zs) > 3).mean()),
                         bright0=im[0][0], sat0=im[0][1], bright_mid=im[1][0], sat_mid=im[1][1]))
    print(f'{nm}: {len(rows)} episodes', flush=True)
    return rows


out = Path(sys.argv[2]); rows = []
for arg in sys.argv[3:]:
    nm, rest = arg.split('=', 1); dpath, _, keepf = rest.partition(':')
    keep = set(json.loads(Path(keepf).read_text())) if keepf else None
    rows += analyse(nm, dpath, keep)
out.write_text(json.dumps(rows))
df = pd.DataFrame(rows)
M = ['dur_s', 'idle_start_s', 'idle_end_s', 'longest_pause_s', 'speed', 'jerk', 'openings', 'track_err', 'z_out', 'z_max', 'zg_out', 'state_z_out', 'bright0', 'sat0']
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
for task in (1, 2, 3):
    x = df[df.task == task]
    print(f'\n== task {task}: median [p10-p90] per source')
    tab = {}
    for src, g in x.groupby('src'):
        tab[f'{src} (n={len(g)})'] = {m: f'{g[m].median():.3g} [{g[m].quantile(.1):.3g}-{g[m].quantile(.9):.3g}]' for m in M}
    print(pd.DataFrame(tab).to_string())
    pub = x[x.src == 'public']; own = x[x.src != 'public']
    flags = {}
    for m in ('dur_s', 'idle_start_s', 'idle_end_s', 'longest_pause_s', 'speed', 'jerk', 'openings', 'track_err', 'z_out', 'state_z_out', 'bright0'):
        hi = pub[m].quantile(.99); lo = pub[m].quantile(.01)
        bad = own[(own[m] > hi) | (own[m] < lo)]
        flags[m] = f'{len(bad)}/{len(own)} outside public p1-p99' + (f': eps {sorted(bad.ep.tolist())[:12]}' if len(bad) else '')
    for k, v in flags.items(): print(f'   {k:16s} {v}')
own = df[df.src != 'public']
if len(own) and own.result.notna().any():
    print('\n== own data by result label (median)')
    print(own.groupby(['task', 'result'])[['dur_s', 'idle_start_s', 'longest_pause_s', 'openings', 'speed', 'z_out']].median().round(3).to_string())
