"""Offline check of harness.Proprio on the 216 official task-3 demos, observed every 0.8 s like the 5080 bridge.
The demos do not react to the instructions, so this checks the rules that read the robot (grasp->sweep, false
brush-drop alarms, the stop) and that nothing crashes; sweep->dump is time-based and cannot be judged here."""
import json, sys
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, '.')
import harness as H, real_split as RS
V = sys.argv[1]
tb = pq.read_table(f'{V}/data/chunk-000/file-000.parquet', columns=['episode_index', 'frame_index', 'observation.state.arm.position',
                   'observation.state.effector.position', 'observation.state.ee.position'])
col = lambda k, n: tb.column(k).combine_chunks().flatten().to_numpy().reshape(-1, n)
EP = tb.column('episode_index').to_numpy(); FR = tb.column('frame_index').to_numpy()
Q, GR, EE = col('observation.state.arm.position', 12), col('observation.state.effector.position', 2), col('observation.state.ee.position', 14)
lab = {json.loads(l)['episode_index']: json.loads(l)['labels'][0]['custom_fields'][0]['value'] for l in open(f'{V}/meta/labels.jsonl')}

class Clk: now = 0.0
class Offline(H.Proprio):
    """No LLM client: the ball counts are fed by the test (all_in_now) instead of background replies."""
    def __init__(self, side):
        H.Scheduler.__init__(self)
        self.subtask, self.pan_side, self.n_side_mismatch, self.llm_complete_t, self.arm_hist = None, side, 0, None, []
        self.n_invalid, self.observe, self.observations = 0, False, []
        self.opened, self.n_hold, self.n_drop, self.phase_t, self.q_dump0, self.d_peak = [False, False], 0, 0, 0.0, None, 0.0
        self.n_total, self.n_reply, self.n_all_in, self.last_counts = 2, 0, 0, None
        self.all_in_now = False
    def start(self, obs, clock, env=None):
        self.go('grasp', clock.now, 'start')
    def on_chunk(self, obs, clock, env=None):
        if self.subtask == 'sweep':          # one synthetic LLM reply per observation
            self.n_all_in = self.n_all_in + 1 if self.all_in_now else 0
        H.LLMScheduler.on_chunk = lambda *a, **k: None   # no background queries offline
        super().on_chunk(obs, clock, env)

g_err, drops, stops, phases_end, crashes = [], 0, [], [], 0
for e in [e for e, i in lab.items() if i.startswith('T3')]:
    ix = np.where(EP == e)[0]; ix = ix[np.argsort(FR[ix])]
    q, gr, ee = Q[ix], GR[ix], EE[ix]
    try: seg = RS.segment(gr, ee)
    except ValueError: continue
    s, c = Offline(seg['pan_arm']), Clk()
    try:
        s.start({}, c)
        for f in range(0, len(ix), 24):                    # 0.8 s at 30 fps
            c.now = f / 30
            s.on_chunk({'images': {'top': None, 'left': None, 'right': None}, 'grippers': tuple(gr[f]), 'arm': q[f]}, c)
            if s.done: stops.append((f - seg['release']) / 30); break
    except Exception as ex:
        crashes += 1; print('crash', e, repr(ex)); continue
    ev = [x for x in s.events if 'subtask' in x]
    t_sweep = next((x['t'] for x in ev if x['subtask'] == 'sweep'), None)
    if t_sweep is not None: g_err.append(t_sweep - seg['b1'] / 30)
    # only drops before the demo puts the tools down count as false alarms (after that the demo, which ignores
    # the instructions, has really let go while the time-based sweep is still running)
    drops += sum(1 for a, b in zip(ev, ev[1:]) if a['subtask'] == 'sweep' and b['subtask'] == 'grasp' and b['t'] < seg['release'] / 30)
    phases_end.append(s.subtask)
print('demos', len(g_err) and len(phases_end), 'crashes', crashes)
print('grasp->sweep minus label: median %.2f  min %.2f  max %.2f s, missed %d' % (np.median(g_err), min(g_err), max(g_err), len(phases_end) - len(g_err)))
print('false brush-drop regressions:', drops)
print('stopped in', len(stops), 'demos; stop minus tool release: median %.2f s' % np.median(stops) if stops else 'never stopped (demos end ~1.4 s after the arms stop)')
from collections import Counter; print('last subtask per demo:', Counter(phases_end))

# ---- "following" simulation: the robot plays the demo segment of whatever subtask the harness says; when a segment
# runs out it holds its last frame (the VLA has finished and waits). Checks dump->home and the stop on prompt-following
# behaviour.
print('--- following simulation')
# the grasp segment plays on 1 s past b1: segment() puts b1 at the first frame within 0.02 of the holding value, and in
# 15 demos the brush gripper still reads 0.39-0.41 one frame earlier (closing); holding that frame forever is not what a
# robot does.
seg_of = {'grasp': ('s0', 'b1h'), 'sweep': ('b1', 'b2'), 'dump': ('b2', 'b3'), 'home': ('b3', 'T')}
dump_pos, n_stop, t_total, timeouts, home_by, slow = [], 0, [], 0, 0, []
for e in [e for e, i in lab.items() if i.startswith('T3')]:
    ix = np.where(EP == e)[0]; ix = ix[np.argsort(FR[ix])]
    q, gr, ee = Q[ix], GR[ix], EE[ix]
    try: seg = RS.segment(gr, ee)
    except ValueError: continue
    seg['s0'], seg['b1h'] = 0, min(seg['b1'] + 30, seg['b2'])
    s, c = Offline(seg['pan_arm']), Clk(); s.start({}, c)
    cur, ptr = 'grasp', 0
    for k in range(int(120 / 0.8)):
        c.now = k * 0.8
        a, b = seg[seg_of[cur][0]], seg[seg_of[cur][1]]
        f = min(a + ptr, b - 1)
        s.all_in_now = cur == 'sweep' and f >= seg['b2'] - 15
        s.on_chunk({'images': {'top': None, 'left': None, 'right': None}, 'grippers': tuple(gr[f]), 'arm': q[f]}, c)
        if s.done:
            n_stop += 1; t_total.append(c.now)
            if c.now > 60: slow.append((e, round(c.now, 1), [(x['t'], x.get('subtask'), x.get('why', '')[:45]) for x in s.events if 'subtask' in x]))
            break
        if s.subtask != cur:
            if cur == 'dump' and s.subtask == 'home':
                dump_pos.append((f - seg['b3']) / 30)
                if any('time' in str(x.get('why', '')) for x in s.events[-1:]): home_by += 1
            cur, ptr = s.subtask, 0
        else:
            ptr += 24
    else:
        timeouts += 1
print('dump->home fired at (frame in the played dump segment - b3): median %.2f p5 %.2f p95 %.2f s' % (
    np.median(dump_pos), np.percentile(dump_pos, 5), np.percentile(dump_pos, 95)))
print('stopped %d / %d, total time median %.1f s (p95 %.1f), hit 120 s %d' % (n_stop, n_stop + timeouts, np.median(t_total), np.percentile(t_total, 95), timeouts))
print('runs over 60 s:', len(slow))
for x in slow[:6]: print('  ', x)
