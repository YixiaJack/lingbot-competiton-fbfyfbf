"""Disk guard for training checkpoints (dev machine). Runs from a loop every 10 minutes; stdlib only.

When the filesystem holding a run's checkpoints has less than MIN_FREE_GB free, delete the resumable DCP parts
(model/, optimizer/) of the OLDEST checkpoints on that filesystem, one at a time, until it is above the threshold.
hf_ckpt (submission weights) and extra_state are always kept. A checkpoint is never touched if it is
  - the newest checkpoint of its run (by step),
  - missing a complete hf_ckpt (config.json + every shard in the index), or written to in the last SETTLE_MIN minutes,
  - in a run whose newest checkpoint is still being written (any file younger than SETTLE_MIN).
Writes a status file (STATUS) for the board.   python3 ckpt_guard.py [--dry-run] [--runs DIR] [--min-free-gb N]"""
import argparse, datetime as dt, glob, json, os, re, shutil, time

ap = argparse.ArgumentParser()
ap.add_argument('--runs', default=os.path.expanduser('~/lingbot_train/runs'))
ap.add_argument('--min-free-gb', type=float, default=float(os.environ.get('MIN_FREE_GB', 150)))
ap.add_argument('--settle-min', type=float, default=10)
ap.add_argument('--status', default=os.path.expanduser('~/lingbot_train/logs/ckpt_guard.json'))
ap.add_argument('--dry-run', action='store_true')
a = ap.parse_args()
now = time.time()
log = lambda *x: print(dt.datetime.now().strftime('%m-%d %H:%M:%S'), *x, flush=True)


def newest_mtime(p):
    t = 0
    for root, _, files in os.walk(p):
        for f in files:
            try: t = max(t, os.lstat(os.path.join(root, f)).st_mtime)
            except OSError: pass
    return t


def hf_complete(c):
    h = os.path.join(c, 'hf_ckpt')
    if not os.path.isfile(os.path.join(h, 'config.json')):
        return False
    idx = os.path.join(h, 'model.safetensors.index.json')
    if os.path.isfile(idx):
        try: shards = set(json.load(open(idx))['weight_map'].values())
        except (ValueError, KeyError): return False
        if not all(os.path.isfile(os.path.join(h, s)) for s in shards): return False
    return now - newest_mtime(h) > a.settle_min * 60


ckpts = []
for run in sorted(glob.glob(os.path.join(a.runs, '*'))):
    cs = []
    for c in glob.glob(os.path.join(run, 'checkpoints', 'global_step_*')):
        m = re.search(r'global_step_(\d+)$', c)
        if m and os.path.isdir(c): cs.append((int(m.group(1)), c))
    if not cs: continue
    cs.sort(); newest = cs[-1][1]
    settled = now - newest_mtime(newest) > a.settle_min * 60
    for step, c in cs:
        real = os.path.realpath(c)
        ckpts.append({'run': os.path.basename(run), 'step': step, 'path': c, 'dev': os.stat(real).st_dev, 'real': real,
                      'dcp': any(os.path.isdir(os.path.join(c, p)) for p in ('model', 'optimizer')),
                      'newest': c == newest, 'run_settled': settled, 'mtime': os.stat(real).st_mtime})
status = {'checked': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), 'min_free_gb': a.min_free_gb,
          'dry_run': a.dry_run, 'filesystems': [], 'actions': []}
for dev in sorted({c['dev'] for c in ckpts}):
    group = [c for c in ckpts if c['dev'] == dev]
    st = os.statvfs(group[0]['real']); free = st.f_bavail * st.f_frsize / 1e9
    mnt = group[0]['real']
    while os.path.dirname(mnt) != mnt and os.stat(os.path.dirname(mnt)).st_dev == dev: mnt = os.path.dirname(mnt)
    fs = {'path': mnt, 'runs': sorted({c['run'] for c in group}), 'free_gb': round(free, 1),
          'total_gb': round(st.f_blocks * st.f_frsize / 1e9, 1), 'pruned': 0}
    if free < a.min_free_gb:
        cand = sorted([c for c in group if c['dcp'] and not c['newest'] and c['run_settled'] and hf_complete(c['path'])],
                      key=lambda c: c['mtime'])
        log(f"{fs['path']}: {free:.0f} GB free < {a.min_free_gb:.0f}, {len(cand)} prunable checkpoint(s)")
        for c in cand:
            if free >= a.min_free_gb: break
            msg = f"{c['run']}/global_step_{c['step']}: remove model/ optimizer/ (hf_ckpt kept)"
            if a.dry_run: log('DRY', msg); free += 49
            else:
                for p in ('model', 'optimizer'): shutil.rmtree(os.path.join(c['path'], p), ignore_errors=True)
                st = os.statvfs(group[0]['real']); free = st.f_bavail * st.f_frsize / 1e9
                log(msg, f'-> {free:.0f} GB free')
            fs['pruned'] += 1
            status['actions'].append({'time': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), 'msg': msg})
        if free < a.min_free_gb: log(f"{fs['path']}: still {free:.0f} GB free, nothing more may be pruned")
    fs['free_gb'] = round(free, 1)
    status['filesystems'].append(fs)
if not a.dry_run:
    old = {}
    try: old = json.load(open(a.status))
    except (OSError, ValueError): pass
    status['actions'] = (old.get('actions', []) + status['actions'])[-20:]
    os.makedirs(os.path.dirname(a.status), exist_ok=True)
    json.dump(status, open(a.status, 'w'))
print(json.dumps({k: status[k] for k in ('min_free_gb', 'filesystems')}, ensure_ascii=False))
