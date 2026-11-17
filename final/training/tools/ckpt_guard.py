"""Disk guard + OSS backup for training checkpoints (dev machine), every 10 minutes from a loop; stdlib only.

1. System disk (the pod's overlay, home dir): the platform evicts the pod at ~300 GB used (11/14 18:55 we lost the pod
   that way). Status 'warn' above SYS_WARN_GB (200); above SYS_PRUNE_GB (240) the DCP parts of older checkpoints that
   sit on the system disk are deleted (runs belong on /data; train_run.sh refuses anything else).
2. /data and any other checkpoint disk: below MIN_FREE_GB (150) free, delete model/ + optimizer/ of the OLDEST
   checkpoints until above it.
   Never touched: the newest checkpoint of a run, a checkpoint whose hf_ckpt is incomplete or written in the last
   SETTLE minutes, any checkpoint of a run whose newest one is still being written; hf_ckpt and extra_state always stay.
3. Backup: every complete, settled hf_ckpt is copied once to OSS (survives the pod), named by the OVERALL step
   (run step + runs/<run>/step_offset), e.g. v4 global_step_1500 -> /mnt/forge/checkpoints/global_step_3500/hf_ckpt,
   the same naming as v3's global_step_1500/2000 and the registered global_step_3000. That path can be registered as is.
   If the name already holds a different checkpoint, it falls back to /mnt/forge/checkpoints/<run>/global_step_N.
   The OSS mount refuses chmod/utime, so only file contents are copied (copytree/copy2/cp -r all fail with EPERM).
   config.json is written last and marks a complete copy; source.txt next to hf_ckpt records which checkpoint it is.
   A complete copy without source.txt (made by hand) counts as ours when the sizes and the last 1 MB of every shard
   match.
Status for the board: ~/lingbot_train/logs/ckpt_guard.json.   python3 ckpt_guard.py [--dry-run]"""
import argparse, datetime as dt, glob, hashlib, json, os, re, shutil, time

ap = argparse.ArgumentParser()
ap.add_argument('--runs', default=os.path.expanduser('~/lingbot_train/runs'))
ap.add_argument('--min-free-gb', type=float, default=float(os.environ.get('MIN_FREE_GB', 150)))
ap.add_argument('--sys-warn-gb', type=float, default=float(os.environ.get('SYS_WARN_GB', 200)))
ap.add_argument('--sys-prune-gb', type=float, default=float(os.environ.get('SYS_PRUNE_GB', 240)))
ap.add_argument('--settle-min', type=float, default=10)
ap.add_argument('--oss', default='/mnt/forge/checkpoints')
ap.add_argument('--status', default=os.path.expanduser('~/lingbot_train/logs/ckpt_guard.json'))
ap.add_argument('--dry-run', action='store_true')
a = ap.parse_args()
now = time.time()
log = lambda *x: print(dt.datetime.now().strftime('%m-%d %H:%M:%S'), *x, flush=True)
HOME_DEV = os.stat(os.path.expanduser('~')).st_dev


def newest_mtime(p):
    t = 0
    for root, _, files in os.walk(p):
        for f in files:
            try: t = max(t, os.lstat(os.path.join(root, f)).st_mtime)
            except OSError: pass
    return t


def hf_complete(c, settle=True):
    h = os.path.join(c, 'hf_ckpt')
    if not os.path.isfile(os.path.join(h, 'config.json')):
        return False
    idx = os.path.join(h, 'model.safetensors.index.json')
    if os.path.isfile(idx):
        try: shards = set(json.load(open(idx))['weight_map'].values())
        except (ValueError, KeyError): return False
        if not all(os.path.isfile(os.path.join(h, s)) for s in shards): return False
    return (not settle) or now - newest_mtime(h) > a.settle_min * 60


def du_gb(p):
    tot = 0
    for root, _, files in os.walk(p):
        for f in files:
            try: tot += os.lstat(os.path.join(root, f)).st_size
            except OSError: pass
    return tot / 1e9


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
        ckpts.append({'run': os.path.basename(run), 'step': step, 'path': c, 'real': real, 'dev': os.stat(real).st_dev,
                      'dcp': any(os.path.isdir(os.path.join(c, p)) for p in ('model', 'optimizer')),
                      'newest': c == newest, 'run_settled': settled, 'mtime': os.stat(real).st_mtime})
status = {'checked': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), 'min_free_gb': a.min_free_gb,
          'sys_warn_gb': a.sys_warn_gb, 'sys_prune_gb': a.sys_prune_gb, 'dry_run': a.dry_run, 'filesystems': [],
          'actions': [], 'published': [], 'system': {}}


def prune(c, why):
    msg = f"{c['run']}/global_step_{c['step']}: remove model/ optimizer/ (hf_ckpt kept) [{why}]"
    if a.dry_run: log('DRY', msg)
    else:
        for p in ('model', 'optimizer'): shutil.rmtree(os.path.join(c['path'], p), ignore_errors=True)
        log(msg)
    status['actions'].append({'time': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), 'msg': msg})


def prunable(group):
    return sorted([c for c in group if c['dcp'] and not c['newest'] and c['run_settled'] and hf_complete(c['path'])],
                  key=lambda c: c['mtime'])


# 1. system disk: judged by USED (eviction limit), not by the filesystem's free space
st = os.statvfs(os.path.expanduser('~'))
sys_used = (st.f_blocks - st.f_bfree) * st.f_frsize / 1e9
sys_state = 'ok' if sys_used < a.sys_warn_gb else 'warn' if sys_used < a.sys_prune_gb else 'prune'
if sys_state == 'prune':
    log(f'SYSTEM DISK {sys_used:.0f} GB used >= {a.sys_prune_gb:.0f}: pruning checkpoints on the system disk')
    for c in prunable([c for c in ckpts if c['dev'] == HOME_DEV]):
        prune(c, 'system disk'); st = os.statvfs(os.path.expanduser('~'))
        sys_used = (st.f_blocks - st.f_bfree) * st.f_frsize / 1e9
        if sys_used < a.sys_prune_gb: break
elif sys_state == 'warn':
    log(f'SYSTEM DISK {sys_used:.0f} GB used >= {a.sys_warn_gb:.0f} (eviction ~300 GB): move or delete large files')
status['system'] = {'used_gb': round(sys_used, 1), 'state': sys_state, 'evict_at_gb': 300}
# 2. checkpoint disks other than the system disk: free space
for dev in sorted({c['dev'] for c in ckpts if c['dev'] != HOME_DEV}):
    group = [c for c in ckpts if c['dev'] == dev]
    s = os.statvfs(group[0]['real']); free = s.f_bavail * s.f_frsize / 1e9
    mnt = group[0]['real']
    while os.path.dirname(mnt) != mnt and os.stat(os.path.dirname(mnt)).st_dev == dev: mnt = os.path.dirname(mnt)
    if free < a.min_free_gb:
        for c in prunable(group):
            if free >= a.min_free_gb: break
            prune(c, f'{mnt} {free:.0f} GB free')
            s = os.statvfs(group[0]['real']); free = s.f_bavail * s.f_frsize / 1e9 if not a.dry_run else free + 49
    status['filesystems'].append({'path': mnt, 'runs': sorted({c['run'] for c in group}), 'free_gb': round(free, 1),
                                  'total_gb': round(s.f_blocks * s.f_frsize / 1e9, 1)})
# 3. OSS backup of every complete, settled hf_ckpt
def tail_sig(h):
    """sha256 over the last 1 MB of every weight shard: tells two checkpoints apart without reading 12 GB from OSS."""
    sig = hashlib.sha256()
    for f in sorted(os.listdir(h)):
        if f.endswith('.safetensors'):
            p = os.path.join(h, f); n = os.path.getsize(p)
            with open(p, 'rb') as fh:
                fh.seek(max(0, n - (1 << 20))); sig.update(f.encode()); sig.update(fh.read())
    return sig.hexdigest()


def same_sizes(src, dst):
    try: return all(os.path.getsize(os.path.join(dst, f)) == os.path.getsize(os.path.join(src, f)) for f in os.listdir(src))
    except OSError: return False


def copy_plain(src, dst):
    """File contents only (no chmod/utime). A file whose size already matches is kept, so an interrupted copy resumes;
    config.json goes last, so its presence marks a complete copy."""
    os.makedirs(dst, exist_ok=True)
    for f in sorted(os.listdir(src), key=lambda f: f == 'config.json'):
        s, d = os.path.join(src, f), os.path.join(dst, f)
        if os.path.isdir(s): raise RuntimeError(f'unexpected subdirectory {s}')
        if f != 'config.json' and os.path.isfile(d) and os.path.getsize(d) == os.path.getsize(s): continue
        shutil.copyfile(s, d)
        if os.path.getsize(d) != os.path.getsize(s): raise RuntimeError(f'size mismatch after copy: {d}')


def read(p):
    try: return open(p).read().strip()
    except OSError: return None


if os.path.isdir(a.oss):
    for c in sorted(ckpts, key=lambda c: (c['run'], c['step'])):
        if not hf_complete(c['path']):
            continue
        src = os.path.join(c['path'], 'hf_ckpt'); me = os.path.realpath(src)
        try: off = int(read(os.path.join(a.runs, c['run'], 'step_offset')) or 0)
        except ValueError: off = 0
        label = f"{c['run']}/global_step_{c['step']}"
        done = target = None
        for base in (os.path.join(a.oss, f"global_step_{c['step'] + off}"), os.path.join(a.oss, c['run'], f"global_step_{c['step']}")):
            dst, owner = os.path.join(base, 'hf_ckpt'), read(os.path.join(base, 'source.txt'))
            if os.path.isfile(os.path.join(dst, 'config.json')):
                if owner == me: done = base; break
                if owner is None and same_sizes(src, dst) and tail_sig(dst) == tail_sig(src):
                    done = base
                    if not a.dry_run: open(os.path.join(base, 'source.txt'), 'w').write(me + '\n')
                    break
                continue                                   # holds another checkpoint: try the next name
            if owner == me or not os.path.exists(dst):
                target = base; break                       # our interrupted copy, or a free name
        if done:
            status['published'].append(f"{label} -> {done}/hf_ckpt"); continue
        if not target:
            log(f'publish {label}: no free OSS name'); continue
        dst = os.path.join(target, 'hf_ckpt')
        if a.dry_run:
            log('DRY publish', src, '->', dst); continue
        t0 = time.time()
        try:
            os.makedirs(target, exist_ok=True); open(os.path.join(target, 'source.txt'), 'w').write(me + '\n')
            copy_plain(src, dst)
            log(f"published {label} hf_ckpt -> {dst} in {time.time() - t0:.0f} s")
            status['published'].append(f"{label} -> {dst}")
        except Exception as e:
            log(f"publish {label} -> {dst} failed: {e!r}")
if not a.dry_run:
    old = {}
    try: old = json.load(open(a.status))
    except (OSError, ValueError): pass
    status['actions'] = (old.get('actions', []) + status['actions'])[-20:]
    os.makedirs(os.path.dirname(a.status), exist_ok=True)
    json.dump(status, open(a.status, 'w'))
print(json.dumps({k: status[k] for k in ('system', 'filesystems', 'published')}, ensure_ascii=False))
