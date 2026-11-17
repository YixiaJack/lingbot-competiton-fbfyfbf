"""Offline replay of TakeoverDetector on the recorded teleop_action timestamps (robot bag 11/15 04:26:29-04:28:24, one
takeover ~04:28:04.4-04:28:17.4) + perturbed copies: bursty delivery (network batching), 30% drops (best effort),
clock jitter. Expect exactly one start ~0.2 s after the takeover and one end ~0.5 s after the handback, nothing else.
usage: python test_takeover_watch.py [teleop_ts.npy]"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from takeover_watch import TakeoverDetector

ts0 = np.load(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent.parent / 'teleop_ts_20261115_0426.npy')
fmt = lambda t: time.strftime('%H:%M:%S', time.localtime(t)) + f'.{int(round((t % 1) * 100)):02d}'
d = np.diff(ts0); fast = np.where(d < 0.012)[0]
t_on, t_off = ts0[fast[0]], ts0[fast[-1] + 1]   # first/last message of the ~226 Hz run
print('takeover in the bag:', fmt(t_on), '->', fmt(t_off))

def replay(ts, period=0.02):
    det, out, i = TakeoverDetector(), [], 0
    for now in np.arange(ts[0], ts[-1] + 2.0, period):
        while i < len(ts) and ts[i] <= now:
            det.feed(ts[i]); i += 1
        ev = det.update(now)
        if ev: out.append((ev, now))
    return out

rng = np.random.default_rng(0)
def burst(ts, b): return np.ceil(ts / b) * b
cases = {'as recorded': ts0,
         'bursty 20 ms': burst(ts0, 0.02),
         'bursty 50 ms': burst(ts0, 0.05),
         '30% dropped': ts0[rng.random(len(ts0)) > 0.3],
         }
j = ts0[rng.random(len(ts0)) > 0.3]; cases['30% dropped + 0-15 ms jitter + 20 ms bursts'] = np.sort(burst(j + rng.random(len(j)) * 0.015, 0.02))
ok = True
for name, ts in cases.items():
    ev = replay(ts)
    good = len(ev) == 2 and ev[0][0] == 'start' and ev[1][0] == 'end' and 0 < ev[0][1] - t_on < 0.5 and 0 < ev[1][1] - t_off < 0.9
    ok &= good
    print(f"{'OK ' if good else 'BAD'} {name:45s}", ', '.join(f'{e} {fmt(t)} ({t - (t_on if e == "start" else t_off):+.2f} s)' for e, t in ev))
# synthetic: a short accidental bump (master moved 0.08 s at 226 Hz) must not trigger; 30 Hz forever must not trigger
syn = np.concatenate([np.arange(0, 5, 1 / 30), np.arange(5, 5.08, 1 / 226), np.arange(5.08, 10, 1 / 30)])
ev = replay(syn); print('OK ' if not ev else 'BAD', '80 ms bump at 226 Hz:', ev or 'no events'); ok &= not ev
print('ALL OK' if ok else 'FAILURES')
