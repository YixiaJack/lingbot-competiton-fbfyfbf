"""Offline checks of the stuck-brush nudge (no LLM calls). usage: python test_nudge.py <code dir>"""
import sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
import harness as H

fails = 0
img = np.zeros((8, 8, 3), np.uint8)


def check(name, cond, extra=''):
    global fails
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{extra}]' if extra else ''))
    fails += not cond


def sim(net_right=0.02, flag=None, subtask='grasp', right_trace=None, T=45.0):
    nf = Path(tempfile.mkdtemp()) / 'nudge'
    H.NUDGE_FILE = nf
    if flag is not None:
        nf.write_text(flag)
    f = H.Fixed()
    f.pan_side, f.subtask, f.current = 'left', subtask, H.RS.instruction(subtask, 'left')
    sweep_txt = H.RS.instruction('sweep', 'left')
    on = []
    t = 0.0
    while t <= T:
        lg = 0.0 if t < 1 else (0.10 if t < 2.5 else 0.032)            # dustpan arm: holds from ~3 s
        rg = right_trace(t) if right_trace else (0.0 if t < 5 else (0.10 if t < 6.5 else 0.06))   # brush stuck at 0.06
        f.net_q = {'left': 0.99, 'right': net_right}
        f.track({'images': {'top': img, 'left': img, 'right': img}, 'grippers': (lg, rg)}, t)
        f.nudge_check(t)
        on.append((round(t, 1), f.prompt() == sweep_txt))
        t += 0.8
    starts = [t for (t, a), (_, b) in zip(on, on[1:] + [(0, False)]) if False] or \
             [on[i][0] for i in range(len(on)) if on[i][1] and (i == 0 or not on[i - 1][1])]
    lens = []
    for st in starts:
        k = [t for t, a in on if t >= st]
        run = 0
        for t, a in on:
            if t >= st:
                if a: run += 1
                else: break
        lens.append(round(run * 0.8, 1))
    return f, starts, lens


f, starts, lens = sim()
check('stuck brush: first nudge 8 s after the stuck state begins', len(starts) >= 1 and 13 <= starts[0] <= 17, f'starts {starts}')
check('nudge lasts about 2.4 s then back to grasp', all(2.0 <= x <= 3.3 for x in lens), f'lengths {lens}')
check('at most 2 nudges, at least 10 s apart', len(starts) == 2 and starts[1] - starts[0] >= 10, f'starts {starts}')
check('nudges recorded in events and summary', sum(1 for e in f.events if 'arm stuck' in str(e.get('nudge', ''))) == 2
      and f.summary()['n_nudge'] == 2)
f, starts, _ = sim(net_right=0.98, right_trace=lambda t: 0.0 if t < 5 else (0.10 if t < 6.5 else 0.023))
check('brush held (network) -> no nudge', starts == [], f'starts {starts}')
f, starts, _ = sim(flag='off')
check('switch file off -> no nudge', starts == [])
f, starts, _ = sim(subtask='sweep')
check('not in grasp -> no nudge', f.n_nudge == 0 and not any('nudge' in e for e in f.events))
f, starts, _ = sim(right_trace=lambda t: 0.0)
check('brush arm never opened (not started) -> no nudge', starts == [])
f, starts, _ = sim(right_trace=lambda t: 0.0 if t < 5 else (0.10 if t < 6.5 else (0.06 if t < 10 else 0.023)), net_right=0.02, T=12)
check('short stall (< 8 s) -> no nudge', starts == [], f'starts {starts}')
# dustpan arm stuck while the brush arm holds (replay ep173 pattern): also nudged
def sim2():
    nf = Path(tempfile.mkdtemp()) / 'nudge'; H.NUDGE_FILE = nf
    f = H.Fixed(); f.pan_side, f.subtask, f.current = 'left', 'grasp', H.RS.instruction('grasp', 'left')
    t = 0.0
    while t <= 30:
        rg = 0.0 if t < 1 else (0.10 if t < 2.5 else 0.023)          # brush arm holds
        lg = 0.0 if t < 4 else (0.10 if t < 5.5 else 0.012)          # dustpan arm closed weakly, not holding
        f.net_q = {'left': 0.03, 'right': 0.99}
        f.track({'images': {'top': img, 'left': img, 'right': img}, 'grippers': (lg, rg)}, t); f.nudge_check(t); t += 0.8
    return f
f = sim2()
check('dustpan arm stuck while the brush arm holds -> nudged too', f.n_nudge >= 1 and any('left arm stuck' in str(e.get('nudge', '')) for e in f.events))
sys.exit(1 if fails else 0)
