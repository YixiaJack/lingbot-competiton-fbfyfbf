"""Action-chunk smoother for the 5080 lingbot_infer service (team patch, 11/15). numpy only.

Why: recorded policy commands carry high-frequency noise inside every chunk. Arm joints reverse direction ~11 times/s
vs 0.3/s for teleop, and the measured gripper chatters 2-2.6 times/s vs 0.04/s. That noise is not at the RTC seams
(VLA-RAIL, arXiv 2512.24673, smooths each chunk at deploy time for the same reason).
What it does to each chunk, in LingBotVLA2Policy.postprocess, before the scheduler and the RTC prefix see it:
  - arm joints: Savitzky-Golay (window 9 steps = 0.3 s, order 2), centred, polynomial fit at the edges. No lag.
    It keeps < 3 Hz motion and removes the 5-15 Hz zig-zag.
  - gripper (0 = closed .. 1 = open, the chunk's units):
    - median over 7 steps removes flickers of <= 3 steps;
    - hysteresis: once closed (< CLOSE), the command must stay > OPEN for OPEN_HOLD steps to open. Until then it
      holds the lowest closed command seen, so thin objects stay squeezed. A real release is delayed by OPEN_HOLD
      steps (0.13 s); short partial openings are ignored.
Per task: config 'tasks' (default [1, 2, 3]; task from the adapter's current instruction, see task_id).
Mode file ~/actsmooth/mode, read on every chunk:
  off    = untouched
  shadow = computed and logged only
  on     = applied
Optional ~/actsmooth/config.json overrides the constants below.
Log: <logs>/actsmooth/<YYYYmmdd_HHMMSS>.jsonl, one line per chunk: residuals and gripper reversals before/after.
Any exception -> the raw chunk is used (never blocks inference)."""
import json, os, time
import numpy as np

HOME = os.path.expanduser('~/actsmooth')
CFG = {'sg_window': 9, 'sg_order': 2, 'grip_median': 7, 'close': 0.2, 'open': 0.5, 'open_hold': 4,
       'arm': True, 'gripper': True, 'session_gap_s': 3.0, 'tasks': [1, 2, 3]}


def task_id(task):
    """3 = sweeping (every task-3 instruction and harness subtask prompt names the dustpan), 1 = plates, 2 = organizer."""
    t = (task or '').lower()
    return 3 if 'dustpan' in t else (1 if 'plate' in t else 2)


def _read(name, default=None):
    try:
        with open(os.path.join(HOME, name)) as f:
            return f.read().strip()
    except OSError:
        return default


def sg_coeffs(window, order):
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(x, order + 1, increasing=True)
    return np.linalg.pinv(A)          # row k: coefficients of x^k; row 0 = smoothing kernel


def savgol(y, window=9, order=2):
    """y: (T, D). Centred Savitzky-Golay; the first/last half-window come from a polynomial fit on the edge window."""
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    if T < window:
        return y.copy()
    P = sg_coeffs(window, order)
    half = window // 2
    out = np.empty_like(y)
    k = P[0]                            # smoothing kernel
    for d in range(y.shape[1]):
        out[half:T - half, d] = np.convolve(y[:, d], k[::-1], mode='valid')
    x = np.arange(-half, half + 1, dtype=np.float64)
    for sl, xs in ((slice(0, window), x[:half]), (slice(T - window, T), x[half + 1:])):
        coef = P @ y[sl]                # (order+1, D) polynomial coefficients on the edge window
        V = np.vander(xs, len(P), increasing=True)
        if sl.start == 0:
            out[:half] = V @ coef
        else:
            out[T - half:] = V @ coef
    return out


def median1d(v, w):
    v = np.asarray(v, dtype=np.float64)
    if w <= 1 or len(v) < w:
        return v.copy()
    h = w // 2
    pad = np.pad(v, (h, h), mode='edge')
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, w), axis=-1)


class GripperGate:
    """Per-arm hysteresis state across chunks of one session."""

    def __init__(self):
        self.closed = None
        self.hold = None

    def step(self, seq, close, open_, open_hold):
        out = seq.copy()
        run = 0
        for i, v in enumerate(seq):
            if self.closed is None:
                self.closed = v < close
                self.hold = v if self.closed else None
            if self.closed:
                if v > open_:
                    run += 1
                    if run >= open_hold:       # sustained opening: release, also release the held steps
                        self.closed, self.hold = False, None
                        out[i] = v
                        continue
                else:
                    run = 0
                self.hold = v if self.hold is None else min(self.hold, v) if v < close else self.hold
                out[i] = min(v, self.hold) if v < open_ else self.hold
            else:
                if v < close:
                    self.closed, self.hold, run = True, v, 0
                out[i] = v
        return out


def reversals(x, eps=1e-3):
    d = np.diff(np.asarray(x, dtype=np.float64), axis=0)
    d[np.abs(d) < eps] = 0
    s = np.sign(d)
    return int(((s[1:] * s[:-1]) < 0).sum())


def hf_residual(y):
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 7:
        return 0.0
    k = np.ones(7) / 7
    sm = np.stack([np.convolve(np.pad(y[:, d], 3, mode='edge'), k, mode='valid') for d in range(y.shape[1])], 1)
    return float(np.sqrt(((y - sm) ** 2).mean()))


class Smoother:
    def __init__(self, log_dir):
        self.log_dir = os.path.join(log_dir, 'actsmooth')
        self.gates = {}
        self.last_t = 0.0
        self.log_path = None
        self.n = 0

    def _new_session(self):
        self.gates = {}
        self.n = 0
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, time.strftime('%Y%m%d_%H%M%S') + '.jsonl')

    def apply(self, actions, arm_keys, grip_keys, task=None):
        """actions: dict key -> (T, dof) float arrays. Returns the dict to execute (smoothed or raw by mode and task)."""
        mode = _read('mode', 'off')
        if mode not in ('shadow', 'on'):
            return actions
        cfg = dict(CFG)
        try:
            cfg.update(json.loads(_read('config.json', '{}') or '{}'))
        except ValueError:
            pass
        if task_id(task) not in cfg['tasks']:   # 11/15 17:40 user: no smoothing on tasks 1 and 2
            return actions
        now = time.monotonic()
        if self.log_path is None or now - self.last_t > cfg['session_gap_s']:
            self._new_session()
        self.last_t = now
        out = dict(actions)
        rec = {'t': round(time.time(), 3), 'mode': mode, 'chunk': self.n}
        for key in arm_keys:
            raw = np.asarray(actions[key], dtype=np.float64)
            sm = savgol(raw, cfg['sg_window'], cfg['sg_order']) if cfg['arm'] else raw
            rec[key.split('.')[1] + '_hf'] = [round(hf_residual(raw), 5), round(hf_residual(sm), 5)]
            rec[key.split('.')[1] + '_maxdev'] = round(float(np.abs(sm - raw).max()), 4)
            out[key] = sm.astype(np.float32)
        for key in grip_keys:
            raw = np.asarray(actions[key], dtype=np.float64)[:, 0]
            g = median1d(raw, cfg['grip_median']) if cfg['gripper'] else raw
            gate = self.gates.setdefault(key, GripperGate())
            if cfg['gripper']:
                g = gate.step(g, cfg['close'], cfg['open'], cfg['open_hold'])
            rec[key.split('.')[1] + '_rev'] = [reversals(raw), reversals(g)]
            rec[key.split('.')[1] + '_range'] = [round(float(raw.min()), 3), round(float(raw.max()), 3),
                                                 round(float(g.min()), 3), round(float(g.max()), 3)]
            out[key] = g[:, None].astype(np.float32)
        self.n += 1
        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')
        except OSError:
            pass
        return out if mode == 'on' else actions
