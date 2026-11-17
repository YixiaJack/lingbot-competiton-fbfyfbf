"""Login-node controller for the whole real-scene experiment: gate files, idempotent training-group submission,
summaries, board config. Never touches jobs outside this experiment's names (lb_real_*). No deadline (decided 11/12);
stops at once if R/PAUSED exists."""
import datetime as dt, fcntl, json, os, subprocess, time
from pathlib import Path

R = Path(os.path.expandvars('$SCRATCH/lingbot/final_real_20261112')); KIT = R.parent / 'tmp/gpu_queue_kit'
RUNS = json.loads((R / 'ab_runs.json').read_text())
LABEL = {'s22_text': '22k+8k → 纯文本', 's22_box': '22k+8k → 画框混训（50%）', 's22_coords': '22k+8k → 坐标混训',
         's22_boxgrasp': '22k+8k → 画框混训，夹爪闭合后去掉目标框（50%）',
         'base_text': '官方 6B 底座 → 纯文本', 'base_box': '官方 6B 底座 → 画框混训'}
lock = open(R / 'controller.lock', 'w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
deadline = float('inf')   # no deadline (decided 11/12): runs until everything is done or R/PAUSED appears
sp = R / 'controller_state.json'
state = json.loads(sp.read_text()) if sp.exists() else {}
(R / 'controller.pid').write_text(str(os.getpid()))
(R / 'gates').mkdir(exist_ok=True)


def sh(args, timeout=180, env=None):
    r = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, env=env)
    print(r.stdout, flush=True)
    if r.returncode: raise RuntimeError(f'{args[:2]} returned {r.returncode}')
    return r.stdout


def save():
    state['updated_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    t = sp.with_suffix('.tmp'); t.write_text(json.dumps(state, indent=1)); t.replace(sp)


def board():
    p = Path.home() / 'lingbot-board/cfg.json'; c = json.loads(p.read_text())
    c['groups'] = [g for g in c['groups'] if not g['name'].startswith('lb_real_')] + [
        {'name': f'lb_real_{n}_1112', 'label': f'{LABEL[n]}（{RUNS[n][2]} 步，真实场景，kit 成员含非抢占卡）',
         'dir': str(R / 'runs' / n), 'log_glob': str(R / 'logs' / f'lb_real_{n}_1112_*.log'),
         'step_regex': r'Step: (\d+)/(\d+) \[', 'ckpt_glob': str(R / 'runs' / n / 'checkpoints/global_step_*')} for n in RUNS]
    t = p.with_suffix('.real.tmp'); t.write_text(json.dumps(c, ensure_ascii=False, indent=2)); t.replace(p)


board(); last_board = 0
while time.time() < deadline:
    if (R / 'PAUSED').exists():
        print('PAUSED present, exit', flush=True); break
    try:
        if len(list((R / "det/items").glob("*.done"))) >= 110 and not (R / 'det/RENDER_READY').exists():
            (R / 'det/RENDER_READY').touch()
        n_ep = len(list((R / 'episodes').glob('*/complete.json'))); state['episodes'] = n_ep
        if n_ep >= 200: (R / 'COLLECTION_READY').touch()   # task 1 only (catalog.N1 = 200, N2 = 0)
        for name, (_, variant, steps, *_) in RUNS.items():
            ready = (R / 'lerobot/text/READY').exists() and (R / 'lerobot' / variant / 'READY').exists()
            if ready and not state.get(f'{name}_submitted'):
                cfgf = R / 'configs' / f'group_{name}.sh'
                reg = R / 'runs' / name / '.handoff/registry.tsv'
                done_m = {l.split('\t')[2] for l in reg.read_text().splitlines()} if reg.exists() else set()
                names = sh(['bash', '-c', 'source "$1"; for x in "${MEMBERS[@]}"; do printf "%s\\n" "${x%%|*}"; done', 'bash', str(cfgf)]).split()
                missing = [m for m in names if m not in done_m]
                if missing:
                    sh(['bash', str(KIT / 'train/submit_group.sh'), str(cfgf)], timeout=900, env=dict(os.environ, ONLY=' '.join(missing)))
                state[f'{name}_submitted'] = True; save()
            if (R / 'runs' / name / 'DONE').exists():
                (R / 'gates' / name).touch()
                if (R / 'det/YOLO_READY').exists(): (R / 'gates' / f'{name}_YOLO').touch()
                if not state.get(f'{name}_cleaned'):
                    ids = sh(['squeue', '-h', '-u', os.environ['USER'], '-n', f'lb_real_{name}_1112', '-t', 'PD', '-o', '%i']).split()
                    if ids: sh(['scancel', *ids])
                    state[f'{name}_cleaned'] = True; save()
        # GPUs: the per-user cap (gpu48: 16) is shared by pool workers and the 4-GPU training members. Once demos,
        # eval seeds and the detector are done the pool has nothing to claim until a run finishes: keep 6 workers.
        seeds = len(list((R / 'eval_seeds').glob('*.json')))
        if (not state.get('pool_shrunk') and (R / 'COLLECTION_READY').exists() and seeds >= 60
                and (R / 'det/YOLO_READY').exists() and not any((R / 'runs' / n / 'DONE').exists() for n in RUNS)):
            ids = sh(['squeue', '-h', '-u', os.environ['USER'], '-n', 'lb_real_work_1112', '-o', '%i %T']).split('\n')
            ids = sorted([l.split()[0] for l in ids if l.strip()], key=lambda x: (x.split('_')[0], int(x.split('_')[1].strip('[]').split('-')[0]) if '_' in x else 0))
            if len(ids) > 6:
                sh(['scancel', *ids[6:]])
            state['pool_shrunk'] = True; save()
        if state.get('pool_shrunk') and not state.get('pool_regrown') and any((R / 'runs' / n / 'DONE').exists() for n in RUNS):
            sh(['bash', str(KIT / 'pool/submit_pool.sh'), str(R / 'configs/pool_general_more.sh')])
            sh(['bash', str(KIT / 'pool/submit_pool.sh'), str(R / 'configs/pool_general_regrow.sh')])
            state['pool_regrown'] = True; save()
        # the first workers were started before the current phase list: once rendering is done, restart them on it
        if (R / 'det/RENDER_READY').exists() and not state.get('old_workers_requeued'):
            try: sh(['scontrol', 'requeue', '22700414'])
            except Exception as e: state['requeue_error'] = repr(e)
            state['old_workers_requeued'] = True; save()
        n_res = len(list((R / 'results').glob('*/*/result.json')))
        if (R / 'det/selection.json').exists() and n_res >= 280:   # 280 A/B episodes (11/13: the 15 GB cap run was dropped)
            (R / 'gates/OURS_DONE').touch()
        # 11/13 (decision): evaluate with as many plain workers as possible. Once both runs are trained and every item of
        # every ready phase is done, cancel the leftover workers instead of letting them sleep on GPUs.
        if all((R / 'runs' / n / 'DONE').exists() for n in RUNS):
            todo = 0
            for line in (R / 'phases.txt').read_text().splitlines():
                if not line.strip() or line.startswith('#'): continue
                f = line.split('|'); name, items, gate = f[0], f[1], (f[3] if len(f) > 3 else '')
                if gate and not Path(gate).exists(): continue
                its = [x.strip() for x in Path(items).read_text().splitlines() if x.strip() and not x.startswith('#')]
                todo += sum(1 for i in its if not (R / 'pool' / name / 'done' / i).exists())
            state['eval_todo'] = todo
            if todo == 0:
                ids = sh(['squeue', '-h', '-u', os.environ['USER'], '-n', 'lb_real_work_1112', '-o', '%i']).split()
                if ids:
                    sh(['scancel', *ids]); state['idle_workers_cancelled'] = state.get('idle_workers_cancelled', 0) + len(ids); save()
        if (R / 'code/summarize_real.py').exists():
            try: sh(['python3', str(R / 'code/summarize_real.py')], timeout=60)
            except Exception as e: state['summary_error'] = repr(e)
        state['results'] = len(list((R / 'results').glob('*/*/result.json')))
        if time.time() - last_board > 120:
            try: sh(['flock', '-n', str(Path.home() / 'lingbot-board/.lock'), 'env', 'FORCE=1', 'bash', str(Path.home() / 'lingbot-board/runner.sh')], timeout=300)
            except Exception as e: state['board_error'] = repr(e)
            last_board = time.time()
        state.pop('last_error', None); save()
    except Exception as e:
        state['last_error'] = repr(e); save(); print(repr(e), flush=True)
    time.sleep(60)
# no deadline: nothing is cancelled automatically; stop the experiment by creating R/PAUSED and scancel by hand
