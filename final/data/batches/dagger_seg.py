"""Human-takeover segments of DAgger recordings (whole inference sessions recorded as one episode).
Per frame: jerk = sum over joints of |2nd difference of action.arm.position|; rolling median over 31 frames.
  human teleop ~0.005-0.010 (and 30-40 % repeated frames); policy ~0.11-0.15 or more. Gripper commands outside [-0.02, 1.5] are policy only.
A human run = rolling jerk < 0.03 and no policy-unit gripper frame, lasting >= 60 frames, after the first policy frame of the episode.
Segment = run start + settle (|action.arm - state.arm| < 0.15 for 15 frames) + 15 frames ... run end - 15 frames (or episode end)."""
import numpy as np
def segments(c, a, s, min_len=60, thr=0.08, k=15):
    T = len(a); d2 = np.abs(np.diff(a, n=2, axis=0)).sum(1); d2 = np.concatenate([[d2[0]], d2, [d2[-1]]])
    pad = np.pad(d2, k, mode='edge'); rm = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(T)])
    gpol = ((c < -0.02) | (c > 1.5)).any(1)
    pol = (rm >= thr) | gpol
    if not pol.any(): return None, rm, gpol            # no policy at all: a normal teleop demo
    first = int(np.argmax(pol)); hum = ~pol; hum[:first] = False
    runs = []; j = first
    while j < T:
        if hum[j]:
            e = j
            while e < T and hum[e]: e += 1
            if e - j >= min_len: runs.append((j, e))
            j = e
        else: j += 1
    gap = np.abs(a - s).max(1); out = []
    for (b, e) in runs:
        st = b
        for i in range(b, max(b, e - 15)):
            if (gap[i:i + 15] < 0.15).all(): st = i; break
        st += 15; en = e if e == T else e - 15
        if en - st >= 45: out.append((int(st), int(en)))
    return out, rm, gpol
