"""Login-node controller for the sweeping-harness experiment. Only writes gate files, submits the training group once,
and summarises; touches no jobs outside this experiment (names lb_sweep_*). Stops at once if R/PAUSED exists.
  collection done (every collect item done or failed, >= MIN_EPISODES kept) -> COLLECTION_READY  (pool runs prep)
  both datasets READY -> submit the training group (once)
  run DONE            -> TRAIN_DONE  (pool runs the eval phase: 5 harnesses x 6 chunks of 5 scenes)
  eval complete       -> summary.json, then exit
"""
import datetime as dt, fcntl, json, os, subprocess, time
from pathlib import Path

R = Path(os.path.expandvars('$SCRATCH/lingbot/sweep_harness_20261114')); KIT = R.parent / 'tmp/gpu_queue_kit'; C = R / 'code'
RUN, STEPS, MIN_EPISODES, ENOUGH = 'sweep_s22', 5000, 120, 160   # start training at 160 kept, or 120 once all tried
HARNESSES = ['none', 'timed', 'oracle', 'v1', 'v2']
lock = open(R / 'controller.lock', 'w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
(R / 'controller.pid').write_text(str(os.getpid()))
sp = R / 'controller_state.json'
state = json.loads(sp.read_text()) if sp.exists() else {}


def log(msg):
    print(f"{dt.datetime.now():%F %T} {msg}", flush=True)


def save():
    state['updated'] = dt.datetime.now(dt.timezone.utc).isoformat()
    t = sp.with_suffix('.tmp'); t.write_text(json.dumps(state, indent=1)); t.replace(sp)


def sh(args, timeout=600, env=None):
    r = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, env=env)
    log(r.stdout[-2000:])
    if r.returncode:
        raise RuntimeError(f'{args[:2]} returned {r.returncode}')
    return r.stdout


def collect_status():
    items = [int(x) for x in (R / 'lists/collect.txt').read_text().split()]
    kept = [i for i in items if (R / 'episodes' / f'{i:04d}' / 'complete.json').exists()]
    done = set(os.listdir(R / 'pool/collect/done')) if (R / 'pool/collect/done').exists() else set()
    failed = [i for i in items if i not in kept and (R / 'episodes' / f'{i:04d}' / 'errors.json').exists()
              and len(json.loads((R / 'episodes' / f'{i:04d}' / 'errors.json').read_text())) >= 16]
    return items, kept, failed, done


(R / 'lists/eval_chunks.txt').write_text(''.join(f'{h}_{c}\n' for c in range(6) for h in HARNESSES))
while True:
    if (R / 'PAUSED').exists():
        log('PAUSED present, exit'); break
    try:
        items, kept, failed, done = collect_status()
        val_items = [int(x) for x in (R / 'lists/val_items.txt').read_text().split()]
        state.update(episodes=len(kept), failed_items=len(failed))
        train_kept = [i for i in kept if i not in val_items]
        if not (R / 'COLLECTION_READY').exists() and (len(train_kept) >= ENOUGH or (len(kept) + len(failed) >= len(items) and len(train_kept) >= MIN_EPISODES)):
            (R / 'lists/train_items.txt').write_text(''.join(f'{i}\n' for i in sorted(train_kept)))   # both datasets use this set
            (R / 'COLLECTION_READY').write_text(f'{len(train_kept)} training episodes\n'); log(f'COLLECTION_READY ({len(kept)} kept, {len(failed)} failed)')
        if not (R / 'VAL_READY').exists() and all(i in kept or i in failed for i in val_items):
            (R / 'VAL_READY').touch(); log('VAL_READY')
        ready = all((R / 'lerobot' / x.strip() / 'READY').exists() for x in (R / 'lists/variants.txt').read_text().split())
        if ready and not state.get('train_submitted'):
            sh(['bash', str(KIT / 'train/submit_group.sh'), str(R / f'configs/group_{RUN}.sh')], timeout=900)
            state['train_submitted'] = True; save(); log('training group submitted')
        # idle workers would hold GPUs against the per-user cap while the 4-GPU training waits: once the datasets, the
        # judgement test and the eval seeds are done, stop the pool; restart it for the evaluation after training
        seeds = len(list((R / 'eval_seeds').glob('*.json')))
        judged = all((R / 'pool/judge/done' / v).exists() for v in ('v1', 'v2'))
        if ready and seeds >= 30 and judged and not state.get('pool_stopped'):
            ids = sh(['squeue', '-h', '-u', os.environ['USER'], '-n', 'lb_sweep_work', '-o', '%i']).split()
            if ids: sh(['scancel', *ids])
            state['pool_stopped'] = True; save(); log('pool stopped until training is done')
        if (R / 'runs' / RUN / 'DONE').exists() and not (R / 'TRAIN_DONE').exists():
            (R / 'TRAIN_DONE').touch(); log('TRAIN_DONE')
            ids = sh(['squeue', '-h', '-u', os.environ['USER'], '-n', 'lb_sweep_train', '-t', 'PD', '-o', '%i']).split()
            if ids: sh(['scancel', *ids])
        if (R / 'TRAIN_DONE').exists() and state.get('pool_stopped') and not state.get('pool_restarted'):
            for conf in ('pool_general.sh', 'pool_dept.sh'):
                sh(['bash', str(KIT / 'pool/submit_pool.sh'), str(R / 'configs' / conf)])
            state['pool_restarted'] = True; save(); log('pool restarted for the evaluation')
        n_res = sum(1 for h in HARNESSES for _ in (R / 'results' / f'{RUN}_{h}').glob('*/result.json'))
        state['eval_results'] = n_res
        # re-summarise whenever a result lands, so the board (projects.json final_sweep) shows partial results
        if n_res and n_res != state.get('summarized'):
            sh(['bash', str(C / 'runtime.sh'), 'sim', 'python', str(C / 'summarize.py'), '--root', str(R), '--run', RUN])
            state['summarized'] = n_res
        if (R / 'TRAIN_DONE').exists() and n_res >= 30 * len(HARNESSES):
            log('evaluation complete'); save(); break
        save()
    except Exception as e:
        log(f'error: {e!r}')
    time.sleep(120)
