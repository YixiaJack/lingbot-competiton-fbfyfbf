"""Validate a corrected dustpan-arm rule for real_split.segment on the official 216 task-3 demos and our own ones.
Old rule: the arm whose gripper opens first takes the dustpan; b1 = the brush arm settles on its tool.
New rule: keep the first-opening arm unless BOTH the arm that finishes its grasp first AND the arm that reaches farthest
forward during the pour say the other arm; b1 = the later of the two settle times (both tools held).
usage: validate_roles.py NAME=V1_DIR[:id_prefix] ..."""
import json, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, str(Path.home() / 'lingbot-final-notes/experiments/sweep_harness/code'))
import real_split as R


def segment_new(gr, ee, open_thr=0.38, min_open=5):
    gr, ee = np.asarray(gr, float), np.asarray(ee, float); T = len(gr)
    opens = [[(s, e) for s, e in R.runs(gr[:, a] > open_thr) if e - s >= min_open] for a in (0, 1)]
    if min(len(o) for o in opens) < 2:
        raise ValueError(f'expected two gripper openings per arm, got {[len(o) for o in opens]}')

    def settled(a, close_at):
        hold = float(np.median(gr[close_at + 10: close_at + 40, a])); j = close_at
        while j < T - 1 and abs(gr[j, a] - hold) > 0.02:
            j += 1
        return j, hold
    st = [settled(a, opens[a][0][1]) for a in (0, 1)]
    release = min(opens[0][-1][0], opens[1][-1][0])
    first_open = int(np.argmin([o[0][0] for o in opens]))
    first_held = int(np.argmin([s[0] for s in st]))
    both = max(s[0] for s in st)
    reach = [float(ee[both:release, 7 * a].max()) if release > both else -1 for a in (0, 1)]
    far = int(np.argmax(reach))
    pan = first_open if (first_held == first_open or far == first_open) else first_held
    brush = 1 - pan
    g_pan, hold_pan = st[pan]; hold_brush = st[brush][1]; b1 = both
    x, z = ee[:, 7 * pan], ee[:, 7 * pan + 2]
    xmax = float(x[b1:release].max()); near = np.where(x[b1:release] >= xmax - 0.04)[0] + b1
    p_in, p_out = int(near[0]), int(near[-1])
    zref = float(np.percentile(z[b1:p_in], 30)) if p_in - b1 > 10 else float(z[b1])
    b2 = p_in
    while b2 > b1 and z[b2] > zref + 0.03:
        b2 -= 1
    b3 = p_out
    if not (0 < b1 < b2 < b3 < release < T):
        raise ValueError(f'boundaries out of order: {b1}, {b2}, {b3}, release {release}, T {T}')
    return {'pan_arm': 'left' if pan == 0 else 'right', 'b1': int(b1), 'b2': int(b2), 'b3': int(b3),
            'grasp_dustpan_done': int(g_pan), 'release': int(release), 'T': T,
            'hold_pan': round(hold_pan, 3), 'hold_brush': round(hold_brush, 3), 'x_pour': round(xmax, 3),
            'votes': {'first_open': first_open, 'first_held': first_held, 'far_reach': far}}


def t3_eps(d, prefix):
    out = []
    for line in open(d / 'meta/labels.jsonl'):
        r = json.loads(line); lb = r['labels'][0]; cf = {f['name']: f['value'] for f in (lb.get('custom_fields') or [])}
        s = lb['task_instruction'].lower()
        if (prefix and cf.get('id', '').startswith(prefix)) or (not prefix and ('dustpan' in s or 'paper ball' in s)):
            out.append(r['episode_index'])
    return out


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        nm, rest = arg.split('=', 1); dpath, _, prefix = rest.partition(':'); d = Path(dpath)
        eps = set(t3_eps(d, prefix))
        import pyarrow as pa; t = pa.concat_tables([pq.read_table(f, columns=['episode_index', 'frame_index', 'observation.state.effector.position', 'observation.state.ee.position'])
                              for f in sorted((d / 'data').glob('*/*.parquet'))])
        ep = t.column('episode_index').to_numpy(); fi = t.column('frame_index').to_numpy()
        G = np.array(t.column('observation.state.effector.position').to_pylist()); E = np.array(t.column('observation.state.ee.position').to_pylist())
        same = changed = fail_both = 0; rows = []
        for e in sorted(eps):
            m = ep == e; o = np.argsort(fi[m]); g, ee = G[m][o], E[m][o]
            try: old = R.segment(g, ee)
            except ValueError as x: old = {'error': str(x)}
            try: new = segment_new(g, ee)
            except ValueError as x: new = {'error': str(x)}
            keys = ('pan_arm', 'b1', 'b2', 'b3', 'release')
            if 'error' in old and 'error' in new: fail_both += 1; continue
            if all(old.get(k) == new.get(k) for k in keys): same += 1
            else:
                changed += 1; rows.append((e, {k: old.get(k, old.get('error')) for k in keys}, {k: new.get(k, new.get('error')) for k in keys + ('x_pour', 'hold_pan', 'hold_brush', 'votes')}))
        print(f'== {nm}: {len(eps)} task-3 demos | identical {same} | changed {changed} | both fail {fail_both}')
        for e, a, b in rows: print(f'   ep{e}: old {a}\n         new {b}')
