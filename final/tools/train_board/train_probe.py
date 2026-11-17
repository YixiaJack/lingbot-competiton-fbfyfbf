"""Runs ON the dev machine (stdlib + optional PyYAML). Reads ~/lingbot_train/runs/* and prints one compact JSON
document as zlib+base64 lines between TRAIN_BEGIN <bytes> <sha256> / TRAIN_END. Read-only."""
import base64, datetime as dt, glob, hashlib, json, os, re, subprocess, time, zlib

HOME = os.path.expanduser('~')
W = os.path.join(HOME, 'lingbot_train')
RUNS = os.path.join(W, 'runs')
STEP_RE = re.compile(r'(\d\d/\d\d/\d{4} \d\d:\d\d:\d\d) - INFO - __main__ - Step (\d+), Epoch (\d+), (.*)$')
BAR_RE = re.compile(r'Step: (\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.]+)s/it\]')
KEYS = ['Loss', 'VLA_Loss', 'Depth_Loss', 'Future_Depth_Loss', 'FutureVideo_Loss', 'SeqWise_Loss', 'MaxVio', 'GradNorm',
        'LR', 'Expert_LR', 'StepTime']
CFG_KEYS = ['lr', 'lr_decay_style', 'micro_batch_size', 'global_batch_size', 'gradient_accumulation_steps', 'max_steps',
            'save_steps', 'train_expert_only', 'freeze_vision_encoder', 'use_moe_expert_lr', 'load_checkpoint_path',
            'model_path', 'action_chunk', 'chunk_size', 'warmup_steps', 'weight_decay']


def read(p, n=None):
    try:
        with open(p, 'rb') as f:
            if n:
                f.seek(0, 2); size = f.tell(); f.seek(max(0, size - n))
            return f.read().decode('utf-8', 'replace')
    except OSError:
        return ''


def alive(pid):
    try:
        cmd = open(f'/proc/{int(pid)}/cmdline', 'rb').read().replace(b'\0', b' ').decode('utf-8', 'replace')
        return 'python' in cmd or 'torchrun' in cmd or 'train' in cmd
    except (OSError, ValueError):
        return False


def to_iso(s):   # '11/14/2026 14:58:31' (dev machine local time, CST) -> ISO with +08:00
    try:
        return dt.datetime.strptime(s, '%m/%d/%Y %H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S+08:00')
    except ValueError:
        return None


def parse_log(text):
    pts = []
    for line in text.replace('\r', '\n').split('\n'):
        m = STEP_RE.search(line)
        if not m:
            continue
        rec = {'s': int(m.group(2)), 't': to_iso(m.group(1)), 'ep': int(m.group(3))}
        for kv in m.group(4).split(','):
            kv = kv.strip().split()
            if len(kv) == 2 and kv[0] in KEYS:
                try: rec[kv[0]] = float(kv[1].rstrip('s'))
                except ValueError: pass
        pts.append(rec)
    bars = BAR_RE.findall(text[-20000:])
    return pts, (bars[-1] if bars else None)


def downsample(pts, n=500, tail=150):
    """Bucket means over the whole run (<= n points) + the last `tail` raw points, so the chart keeps its shape and
    the recent end stays exact."""
    if len(pts) <= n + tail:
        return pts
    head, last = pts[:-tail], pts[-tail:]
    size = len(head) / n; out = []
    for i in range(n):
        b = head[int(i * size):int((i + 1) * size)] or head[int(i * size):int(i * size) + 1]
        rec = {'s': b[-1]['s'], 't': b[-1]['t'], 'ep': b[-1]['ep']}
        for k in KEYS:
            v = [x[k] for x in b if k in x]
            if v: rec[k] = sum(v) / len(v)
        out.append(rec)
    return out + last


def cfg_summary(path):
    out = {}
    try:
        import yaml
        def walk(d, pre=''):
            for k, v in d.items():
                if isinstance(v, dict): walk(v, pre + k + '.')
                elif k in CFG_KEYS and (pre + k) not in out: out[pre + k] = v
        walk(yaml.safe_load(open(path)) or {})
    except Exception:
        for line in read(path).split('\n'):
            m = re.match(r'\s*(\w+):\s*(\S.*)$', line)
            if m and m.group(1) in CFG_KEYS: out.setdefault(m.group(1), m.group(2).strip())
    return {k: (v if isinstance(v, (int, float, bool)) or v is None else str(v)[:120]) for k, v in out.items()}


def dir_size(p):
    tot = 0
    for root, _, files in os.walk(p):
        for f in files:
            try: tot += os.path.getsize(os.path.join(root, f))
            except OSError: pass
    return tot


def run_info(d, full):
    name = os.path.basename(d)
    launch = read(os.path.join(d, 'launch.txt')).strip()
    pid = read(os.path.join(d, 'pid')).strip()
    logp = os.path.join(d, 'train.log')
    text = read(logp) if full else read(logp, 400000)
    pts, bar = parse_log(text)
    tail = read(logp, 6000).replace('\r', '\n')
    err = None
    for line in reversed([l for l in tail.split('\n') if l.strip()][-60:]):
        if re.search(r'(Error|Exception|Traceback|CUDA out of memory|NCCL)', line) and 'Ignore_Batch' not in line:
            err = line.strip()[-300:]; break
    ck = []
    for c in sorted(glob.glob(os.path.join(d, 'checkpoints', '*')), key=os.path.getmtime):
        m = re.search(r'(\d+)$', c)
        ck.append({'name': os.path.basename(c), 'step': int(m.group(1)) if m else None,
                   'parts': sorted(x for x in os.listdir(c) if os.path.isdir(os.path.join(c, x))),
                   'time': dt.datetime.fromtimestamp(os.path.getmtime(c), dt.timezone.utc).isoformat(timespec='seconds'),
                   'bytes': dir_size(c) if full else None})
    m = re.search(r'CFG=(\S+)', launch)
    cfgp = m.group(1) if m else None
    cfg = cfg_summary(cfgp) if (full and cfgp and os.path.exists(cfgp)) else {}
    mstep = None
    for k, v in cfg.items():
        if k.endswith('max_steps'): mstep = int(v)
    if bar: mstep = mstep or int(bar[1])
    last = pts[-1] if pts else None
    info = {'name': name, 'launch': launch, 'started': (re.match(r'(\S+)', launch).group(1) if launch else None),
            'pid': pid, 'alive': alive(pid) if pid else False,
            'log_mtime': (dt.datetime.fromtimestamp(os.path.getmtime(logp), dt.timezone.utc).isoformat(timespec='seconds')
                          if os.path.exists(logp) else None),
            'log_age_s': (round(time.time() - os.path.getmtime(logp)) if os.path.exists(logp) else None),
            'step': last['s'] if last else (int(bar[0]) if bar else 0), 'max_steps': mstep, 'n_points': len(pts),
            'last': last, 'bar': ({'step': int(bar[0]), 'total': int(bar[1]), 'elapsed': bar[2], 'eta': bar[3],
                                   'sit': float(bar[4])} if bar else None),
            'error': err, 'signal15': 'got signal: 15' in tail, 'checkpoints': ck}
    if full:
        info['config'] = cfg; info['config_path'] = cfgp
        info['commit'] = read(os.path.join(d, 'code_commit.txt')).strip()[:300]
        info['env'] = read(os.path.join(d, 'env.txt')).strip()[:300]
        roots = []
        for line in read(os.path.join(d, os.path.basename(d) + '.txt')).split('\n'):
            parts = line.split()
            if len(parts) == 2:
                meta = {}
                try: meta = json.load(open(os.path.join(parts[1], 'meta', 'info.json')))
                except Exception: pass
                roots.append({'robot': parts[0], 'path': parts[1], 'name': os.path.basename(parts[1]),
                              'episodes': meta.get('total_episodes'), 'frames': meta.get('total_frames')})
        info['roots'] = roots
        info['series'] = downsample(pts)
        st = [p['StepTime'] for p in pts[-50:] if 'StepTime' in p]
        info['step_time_recent'] = sorted(st)[len(st) // 2] if st else None
    return info


def gpus():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    res = []
    for line in out.strip().split('\n'):
        p = [x.strip() for x in line.split(',')]
        if len(p) == 7:
            def num(x):
                try: return float(x)
                except ValueError: return None
            res.append({'i': int(p[0]), 'name': p[1], 'util': num(p[2]), 'mem': num(p[3]), 'mem_total': num(p[4]),
                        'temp': num(p[5]), 'power': num(p[6])})
    return res


def main():
    dirs = [d for d in glob.glob(os.path.join(RUNS, '*')) if os.path.isdir(d) and os.path.exists(os.path.join(d, 'train.log'))]
    dirs.sort(key=lambda d: os.path.getmtime(os.path.join(d, 'train.log')))
    live = [d for d in dirs if alive(read(os.path.join(d, 'pid')).strip() or 0)]
    current = (live or dirs[-1:] or [None])[-1]
    runs = [run_info(d, d == current) for d in dirs]
    st = os.statvfs(os.path.realpath(current) if current else HOME); sh = os.statvfs(HOME)
    doc = {'generated': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
           'host': os.uname().nodename, 'current': os.path.basename(current) if current else None,
           'runs': runs, 'gpus': gpus(),
           'train_procs': sum(1 for p in glob.glob('/proc/[0-9]*/cmdline')
                              if b'train_lingbotvla' in (open(p, 'rb').read() if os.access(p, os.R_OK) else b'')),
           'disk': {'free': st.f_bavail * st.f_frsize, 'total': st.f_blocks * st.f_frsize,
                    'path': os.path.realpath(current) if current else HOME},
           'disk_home': {'free': sh.f_bavail * sh.f_frsize, 'total': sh.f_blocks * sh.f_frsize},
           'ab_summary': read(os.path.join(RUNS, 'ab_summary.txt'))[-4000:]}
    g = read(os.path.join(W, 'logs', 'ckpt_guard.json'))
    if g:
        try:
            doc['guard'] = json.loads(g)
            gp = read(os.path.join(W, 'logs', 'ckpt_guard.pid')).strip()
            doc['guard']['alive'] = bool(gp) and os.path.exists(f'/proc/{gp}')
        except ValueError:
            pass
    fs = read(os.path.join(HOME, 'field_training', 'status.json'))
    if fs:
        try:
            s = json.loads(fs)
            doc['field_prep'] = {'state': s.get('state'), 'generated': s.get('generated'),
                                 'private_episodes': s.get('private_episodes'),
                                 'reports': [{'source': r.get('source', '').split('/private/')[-1][:60],
                                              'input': r.get('input_episodes'), 'accepted': r.get('accepted'),
                                              'excluded': r.get('excluded'), 'reasons': r.get('excluded_reasons'),
                                              'task_counts': r.get('task_counts')} for r in s.get('reports', [])]}
        except ValueError:
            pass
    raw = json.dumps(doc, separators=(',', ':'), ensure_ascii=False).encode()
    z = zlib.compress(raw, 9); b = base64.b64encode(z).decode()
    print(f'TRAIN_BEGIN {len(z)} {hashlib.sha256(z).hexdigest()}', flush=True)
    for i in range(0, len(b), 160):
        print(b[i:i + 160])
    print('TRAIN_END', flush=True)


if __name__ == '__main__':
    main()
