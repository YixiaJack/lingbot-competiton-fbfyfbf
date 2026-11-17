"""Offline checks of GripperTrack and the fused grasp rule (no LLM calls). usage: python test_fused.py <code dir>"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import harness as H

fails = 0


def check(name, cond, extra=''):
    global fails
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{extra}]' if extra else ''))
    fails += not cond


def run_track(seq, dt=0.8):
    g = H.GripperTrack()
    out = []
    for i, v in enumerate(seq):
        g.update(i * dt, v)
        out.append(g.snapshot(i * dt))
    return out


# raw metres as the 5080 bridge passes them; 02:07 run left/right gripper pattern
left_0207 = [0, 0, 0, 0, 0, 0, 0, 0.10, 0.10, 0.10, 0.06, 0.03, 0.03, 0.03, 0.03, 0.03]
s = run_track(left_0207)
check('02:07 left: closed and steady above zero at the end', s[-1]['state'] == 'closed' and s[-1]['above_zero'],
      H.GripperTrack.words(s[-1]))
check('02:07 left: not opened during the first zeros', s[3]['state'] == 'not_opened')
check('02:07 left: opening/open while at 0.10', s[8]['state'] in ('open', 'opening'), s[8]['state'])
s = run_track([0, 0, 0.10, 0.10, 0.0, 0.0, 0.0])
check('thin handle / empty: fully closed, not above zero', s[-1]['state'] == 'closed' and not s[-1]['above_zero'],
      H.GripperTrack.words(s[-1]))
s = run_track([0, 0, 0.10, 0.10, 0.045, 0.045, 0.045])
check('stuck just under half (0.045 raw): closed above zero (vision must decide)', s[-1]['state'] == 'closed' and s[-1]['above_zero'])
s = run_track([0, 0, 0.10, 0.10, 0.072, 0.072, 0.072])
check('stopped at 0.72 after 1.0 (ep175 left): counts as open, not closed', s[-1]['state'] == 'open', H.GripperTrack.words(s[-1]))
s = run_track([0, 0, 0.10, 0.10, 0.06, 0.06, 0.06])
check('stopped at 0.6 after 1.0 (fingertips on the table): stopped halfway, not closed', s[-1]['state'] == 'partly', H.GripperTrack.words(s[-1]))
s = run_track([0, 0.10, 0.10, 0.072, 0.072, 0.072, 0.039, 0.0, 0.0, 0.051, 0.028, 0.028, 0.028])
check('ep175 left later: halfway, then closes to 0, reopens a bit, closes to 0.28 -> closed (re-judged)', s[-1]['state'] == 'closed'
      and s[-1]['steady_s'] < 3, H.GripperTrack.words(s[-1]))
s = run_track([0, 0.10, 0.10, 0.08, 0.06, 0.04])
check('still closing: not closed yet', s[-1]['state'] == 'closing', s[-1]['state'])
s = run_track(left_0207[::2], dt=1.6)
check('sparse sampling (1.6 s): still reaches closed and steady', s[-1]['state'] == 'closed', H.GripperTrack.words(s[-1]))
s = run_track([0, 0, 0.007, 0.022, 0.022, 0.022], dt=1.6)
check('opening missed between samples (demo 154 left: 0 -> 0.07 -> 0.22 steady): closed above zero', s[-1]['state'] == 'closed'
      and s[-1]['above_zero'], H.GripperTrack.words(s[-1]))
s = run_track([0, 0, 0.004, 0.004, 0.004])
check('tiny drift at the start (0.04 normalized) stays not opened', s[-1]['state'] == 'not_opened', s[-1]['state'])
s = run_track([0.0] * 8)
check('never opened: not_opened', s[-1]['state'] == 'not_opened')
s = run_track([0, 0.10, 0.10, 0.03, 0.03, 0.03, 0.09, 0.10])
check('closed then opens again (regrasp/release): not closed', s[-1]['state'] in ('opening', 'open'), s[-1]['state'])
s = run_track([0, 0, 1.0, 1.0, 0.32, 0.32, 0.32])
check('0-1 dataset units: closed above zero', s[-1]['state'] == 'closed' and s[-1]['above_zero'])
s = run_track([0, 0.10, 0.10, 0.03, 0.031, 0.029, 0.03])
check('small jitter while holding stays closed', s[-1]['state'] == 'closed')

# fused rule inside Fixed (no LLM calls: user_turn + apply directly)
img = np.zeros((8, 8, 3), np.uint8)


class Clk:
    now = 0.0


def run_fixed(steps):
    """steps: (left_raw, right_raw, reply dict or None); one step = 0.8 s, a reply every second step."""
    f = H.Fixed(); c = Clk(); out = []; first = True
    for i, (lg, rg, rep) in enumerate(steps):
        c.now = i * 0.8
        obs = {'images': {'top': img, 'left': img, 'right': img}, 'grippers': (lg, rg)}
        content, text = f.user_turn(obs, c, first)
        if rep is not None:
            d = {'observation': 'x', 'dustpan_side': 'right', 'task_complete': False, **rep}
            f.apply(d, c.now, {})
            first = False
            out.append((round(c.now, 1), f.subtask, text))
    return f, out


G = {'subtask': 'grasp'}
held = {'subtask': 'grasp', 'left_holds': 'brush', 'right_holds': 'dustpan'}   # dustpan on the right
seq = ([(0, 0, {**G, 'left_holds': 'nothing', 'right_holds': 'nothing'})] + [(0, 0, None), (0, 0.10, G), (0, 0.10, None),
       (0.10, 0.03, G), (0.10, 0.03, None), (0.03, 0.03, G), (0.03, 0.03, None)]
       + [(0.03, 0.03, held), (0.03, 0.03, None)] * 4)
f, out = run_fixed(seq)
subs = [o[1] for o in out]
check('both seen holding + both grippers closed and steady on 2 replies -> sweep while the LLM says grasp',
      subs[5] == 'sweep' and f.n_grasp_override == 1, str(subs))
check('after the switch, the LLM saying grasp while both are still seen held + closed -> stay in sweep',
      subs[5:] == ['sweep'] * 3, str(subs))
lost = {'subtask': 'grasp', 'left_holds': 'nothing', 'right_holds': 'dustpan'}
fl, outl = run_fixed(seq + [(0.03, 0.03, lost)])
check('in sweep, the brush arm seen holding nothing and the LLM says grasp -> back to grasp', outl[-1][1] == 'grasp')
fr, outr = run_fixed(seq + [(0.10, 0.03, None), (0.10, 0.03, held)])
check('in sweep, a gripper reopened and the LLM says grasp -> back to grasp', outr[-1][1] == 'grasp')
print('   gripper words sent to the LLM:', out[-1][2][:200])
check('log line keeps the raw numbers for the viewer', '[raw left 0.030, right 0.030]' in out[-1][2])
check('LLM text has no raw numbers', 'raw' not in f.chat.messages[-1]['content'][0]['text'] if len(f.chat.messages) > 1 else True)

# vision says held but a gripper is still closing -> no switch
seq2 = [(0, 0, G), (0, 0.10, None), (0.10, 0.10, G), (0.08, 0.06, None), (0.06, 0.04, held), (0.05, 0.03, None), (0.04, 0.03, held)]
f2, out2 = run_fixed(seq2)
check('seen holding but grippers still closing -> stay in grasp', all(o[1] == 'grasp' for o in out2), str([o[1] for o in out2]))

# grippers closed and steady but vision says nothing / wrong tool -> no switch (fingertips on the table)
wrong = {'subtask': 'grasp', 'left_holds': 'nothing', 'right_holds': 'dustpan'}
seq3 = [(0, 0, G), (0.10, 0.10, None), (0.06, 0.03, G), (0.06, 0.03, None)] + [(0.06, 0.03, wrong), (0.06, 0.03, None)] * 4
f3, out3 = run_fixed(seq3)
check('grippers closed but vision says the brush arm holds nothing -> stay in grasp', all(o[1] == 'grasp' for o in out3))
swapped = {'subtask': 'grasp', 'left_holds': 'dustpan', 'right_holds': 'brush'}
seq4 = [(0, 0, G), (0.10, 0.10, None), (0.03, 0.03, G), (0.03, 0.03, None)] + [(0.03, 0.03, swapped), (0.03, 0.03, None)] * 3
f4, out4 = run_fixed(seq4)
check('vision says each arm holds the other tool -> stay in grasp', all(o[1] == 'grasp' for o in out4))

# one held reply, then unclear -> streak resets
seq5 = [(0, 0, G), (0.10, 0.10, None), (0.03, 0.03, G), (0.03, 0.03, None), (0.03, 0.03, held), (0.03, 0.03, None),
        (0.03, 0.03, {**G, 'left_holds': 'unclear', 'right_holds': 'dustpan'}), (0.03, 0.03, None), (0.03, 0.03, held)]
f5, out5 = run_fixed(seq5)
check('held, unclear, held -> no switch yet (needs 2 in a row)', all(o[1] == 'grasp' for o in out5))

# the LLM says sweep while a gripper is open -> wait; closing -> allowed
seq8 = [(0, 0, G), (0.03, 0.10, None), (0.03, 0.10, G), (0.03, 0.10, None), (0.03, 0.10, {'subtask': 'sweep'})]
f8, out8 = run_fixed(seq8)
check('LLM says sweep while the right gripper is open -> stay in grasp', out8[-1][1] == 'grasp', str([o[1] for o in out8]))
seq9 = [(0, 0, G), (0.10, 0.10, None), (0.03, 0.10, G), (0.03, 0.07, None), (0.03, 0.05, {'subtask': 'sweep'})]
f9, out9 = run_fixed(seq9)
check('LLM says sweep while the right gripper is closing -> sweep', out9[-1][1] == 'sweep', str([o[1] for o in out9]))

# the LLM's own sweep answer still switches at once
seq6 = [(0, 0, G), (0.10, 0.10, None), (0.03, 0.03, {'subtask': 'sweep'})]
f6, out6 = run_fixed(seq6)
check("LLM says sweep -> sweep at once (unchanged)", out6[-1][1] == 'sweep')

# replies without the new fields (older behaviour) -> no override
seq7 = [(0, 0, G), (0.10, 0.10, None)] + [(0.03, 0.03, G), (0.03, 0.03, None)] * 4
f7, out7 = run_fixed(seq7)
check('no left_holds/right_holds in replies -> LLM decides', all(o[1] == 'grasp' for o in out7) and f7.n_grasp_override == 0)
# --- grasp network (probs injected) ---
if hasattr(H, 'GraspNet'):
    NETP = {'v': [0.99, 0.99]}
    H.GraspNet.probs = classmethod(lambda cls, imgs: list(NETP['v']))
    NETP['v'] = [0.99, 0.98]
    fn, outn = run_fixed(seq)          # LLM says grasp with holds fields; net says both hold, grippers closed
    check('net: both held + grippers closed on 2 replies -> sweep', [o[1] for o in outn][5] == 'sweep', str([o[1] for o in outn]))
    check('net: user text carries the grasp check and the log the probabilities',
          'Grasp check from the wrist cameras: left holds a handle, right holds a handle.' in outn[-1][2] and '[net left 0.99, right 0.98]' in outn[-1][2])
    NETP['v'] = [0.99, 0.02]
    fn2, outn2 = run_fixed(seq)
    check('net: right not holding -> no rule switch even if the LLM fields say held', all(o[1] == 'grasp' for o in outn2))
    sw = {'subtask': 'sweep'}
    seqw = [(0, 0, G), (0.10, 0.10, None), (0.03, 0.03, G), (0.03, 0.03, None)] + [(0.03, 0.03, sw), (0.03, 0.03, None)] * 6
    fw, outw = run_fixed(seqw)
    subs = [o[1] for o in outw]
    check('net says right not holding: LLM sweep held back 4 times, then let through', subs[2:6] == ['grasp'] * 4 and subs[6] == 'sweep', str(subs))
    NETP['v'] = [0.99, 0.97]
    fw2, outw2 = run_fixed(seqw)
    check('net says both hold: LLM sweep goes through at once', [o[1] for o in outw2][2] == 'sweep', str([o[1] for o in outw2]))
    H.GraspNet.probs = classmethod(lambda cls, imgs: None)
    fz, outz = run_fixed(seq)
    check('net unavailable -> falls back to the LLM fields (switch as before)', [o[1] for o in outz][5] == 'sweep')
sys.exit(1 if fails else 0)
