"""Summarize the replica job: timing cases T0-T3 (steps >= 8), restart cases R1/R2, open-loop R3. stdlib only."""
import glob, json, os, re, statistics as st
R = os.path.expandvars('$SCRATCH/lingbot/devrep_20261114')
STEP = re.compile(r'Step (\d+)(?:/\d+)?, Epoch (\d+), (.*)$')
def parse(path):
    pts = []
    if not os.path.exists(path): return pts
    for line in open(path, errors='replace').read().replace('\r', '\n').split('\n'):
        m = STEP.search(line)
        if m and 'StepTime' in line:
            rec = {'s': int(m.group(1))}
            for kv in m.group(3).split(','):
                kv = kv.strip().split()
                if len(kv) == 2:
                    try: rec[kv[0]] = float(kv[1].rstrip('s'))
                    except ValueError: pass
            pts.append(rec)
    return pts
def tail_err(path):
    if not os.path.exists(path): return 'no log'
    lines = [l for l in open(path, errors='replace').read().replace('\r', '\n').split('\n') if l.strip()]
    errs = [l for l in lines[-200:] if re.search(r'(Error|Traceback|Exception|OOM|out of memory)', l)]
    return (errs[-1] if errs else (lines[-1] if lines else 'empty log'))[-300:]
out = {'job': open(f'{R}/results/job_info.txt').read().strip() if os.path.exists(f'{R}/results/job_info.txt') else None}
print(out['job'])
print('\n== timing (steps >= 8; teacher = 2 x Depth_Forward_Time, synced; grad accumulation 2)')
base = None
for case, label in [('T0', 'baseline (dev v2 settings)'), ('T1', '+ built-in color aug'), ('T2', '+ our whitelist aug'),
                    ('T3', 'no teacher alignment')]:
    p = [x for x in parse(f'{R}/runs/{case}/train.log') if x['s'] >= 8]
    if not p:
        print(f'{case} {label}: no steps ({tail_err(f"{R}/runs/{case}/train.log")})'); continue
    stp = st.median([x['StepTime'] for x in p]); dft = st.median([x.get('Depth_Forward_Time', 0) for x in p])
    if case == 'T0': base = stp
    rel = f'{(stp / base - 1) * 100:+.1f}% vs T0' if base and case != 'T0' else ''
    print(f'{case} {label}: step {stp:.3f}s (n={len(p)}), teacher {2 * dft:.3f}s/step = {2 * dft / stp * 100:.1f}% {rel}')
    out[case] = {'step_s': stp, 'teacher_s': 2 * dft, 'n': len(p), 'loss_last': p[-1].get('Loss')}
prof = f'{R}/runs/T0/profile.txt'
if os.path.exists(prof):
    print('\n== T0 profiler (steps 18-19, top CUDA ops)'); print('\n'.join(open(prof).read().split('\n')[:24]))
print('\n== restart path')
for case in ('R1', 'R2'):
    p = parse(f'{R}/runs/{case}/train.log')
    print(f'{case}: steps {[x["s"] for x in p]}  loss {[round(x.get("Loss", 0), 4) for x in p]}  LR {sorted({x.get("LR") for x in p})}'
          if p else f'{case}: no steps ({tail_err(f"{R}/runs/{case}/train.log")})')
    log = open(f'{R}/runs/{case}/train.log', errors='replace').read() if os.path.exists(f'{R}/runs/{case}/train.log') else ''
    for pat in ('Load distributed checkpoint', 'forge-slim', 'Total training steps', 'train_steps'):
        hits = sorted(set(l.strip()[-160:] for l in log.replace('\r', '\n').split('\n') if pat in l))[:4]
        for h in hits: print('   ', h)
if os.path.exists(f'{R}/results/ckpt_dirs.txt'): print(open(f'{R}/results/ckpt_dirs.txt').read())
print('== R3 open-loop:', tail_err(f'{R}/runs/R3/eval.log'))
for f in sorted(glob.glob(f'{R}/runs/R3/*'))[:6]: print('   ', os.path.basename(f))
json.dump(out, open(f'{R}/results/summary.json', 'w'), indent=1)
