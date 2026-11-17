"""Runs ON the 5080 (fed over ssh stdin: `ssh lbreal python3 -u - < follow_remote.py`). Read-only.
Follows the inference logs and prints one JSON line per record:
  {"src": "harness"|"yolocue"|"session"|"hb", "file": ..., "rec": {...}}
- harness: logs/robotharness_*.jsonl (task-3 prompts, LLM replies)      newest 4 files sent in full at start
- yolocue: logs/yolocue/*.jsonl (task-1/2 box cue, grasp, end)          newest 6 files sent in full at start
- session: logs/takeover_events.jsonl (start / pause / resume / stop)  last 300 lines at start
- hb every 2 s: 5080 wall time + harness mode/scheduler + yolocue mode
Never writes anything; polls every 0.3 s."""
import glob, json, os, sys, time

R = os.path.expanduser('~/workspace/lingbot/inference/logs')
TK = R + '/takeover_events.jsonl'
pos = {}


def emit(src, f, line):
    line = line.strip()
    if not line:
        return
    try:
        rec = json.loads(line)
    except ValueError:
        return
    sys.stdout.write(json.dumps({'src': src, 'file': os.path.basename(f), 'rec': rec}, ensure_ascii=False) + '\n')


def follow(src, f, tail_lines=None):
    try:
        size = os.path.getsize(f)
    except OSError:
        return
    p = pos.get(f)
    if p is None and tail_lines:
        with open(f, 'rb') as fh:
            fh.seek(max(0, size - 400000))
            data = fh.read(size - max(0, size - 400000))
        cut = data.rfind(b'\n')
        for l in data[:cut + 1].decode('utf-8', 'ignore').splitlines()[-tail_lines:]:
            emit(src, f, l)
        pos[f] = max(0, size - 400000) + cut + 1
        return
    p = p or 0
    if size < p:
        p = 0
    if size == p:
        pos[f] = p
        return
    with open(f, 'rb') as fh:
        fh.seek(p)
        data = fh.read(size - p)
    cut = data.rfind(b'\n')
    if cut < 0:
        pos[f] = p
        return
    for l in data[:cut].decode('utf-8', 'ignore').splitlines():
        emit(src, f, l)
    pos[f] = p + cut + 1


def read(p):
    try:
        return open(os.path.expanduser(p)).read().strip()
    except OSError:
        return None


def newest(pattern):
    fs = glob.glob(pattern)
    return sorted(fs, key=lambda f: os.path.getmtime(f))


def llm_model():
    try:
        c = json.load(open(os.path.expanduser('~/robotharness/llm.json')))
        return f"{c.get('model')} ({c.get('provider', '?')})"
    except (OSError, ValueError):
        return 'deepseek-flash (DeepSeek)'


first, last_hb = True, 0.0
while True:
    hs, ys, acs = newest(R + '/robotharness_*.jsonl'), newest(R + '/yolocue/*.jsonl'), newest(R + '/actsmooth/*.jsonl')
    if first:
        for f in hs[:-4] + ys[:-6] + acs[:-4]:
            pos[f] = os.path.getsize(f)          # older sessions: skip
        follow('session', TK, tail_lines=300)
        first = False
    else:
        follow('session', TK)
    for f in hs[-6:]:
        follow('harness', f)
    for f in ys[-8:]:
        follow('yolocue', f)
    for f in acs[-4:]:
        follow('actsmooth', f)
    now = time.time()
    if now - last_hb >= 2:
        last_hb = now
        sys.stdout.write(json.dumps({'src': 'hb', 'rec': {'wall_time': now, 'harness_mode': read('~/robotharness/mode'),
                                                          'scheduler': read('~/robotharness/scheduler'),
                                                          'yolocue_mode': read('~/yolocue/mode'),
                                                          'actsmooth_mode': read('~/actsmooth/mode'),
                                                          'llm': llm_model()}}) + '\n')
    sys.stdout.flush()
    time.sleep(0.3)
