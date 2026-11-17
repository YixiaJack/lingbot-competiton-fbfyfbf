"""Bounded experiment controller: gates, idempotent submission, summaries, existing private board.

Runs on the login node; all rendering, conversion, training, inference run in Slurm.
Never touches jobs outside this experiment's exact names.
"""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

R=Path(os.path.expandvars('$SCRATCH/lingbot/final_yolo_20261111'))
KIT=R.parent/'tmp/gpu_queue_kit'
NAMES=['lb_yolo_work_1111']+[f'lb_yolo_{m}_1111' for m in ('text','box','coords')]
lock=open(R/'controller.lock','w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
deadline=dt.datetime(2026,11,13,10,tzinfo=dt.timezone.utc).timestamp()
state_path=R/'controller_state.json'
state=json.loads(state_path.read_text()) if state_path.exists() else {}
state.pop('blocked',None)
(R/'controller.pid').write_text(str(os.getpid()))


def command(args,timeout=180,env=None):
    r=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout,env=env)
    print(r.stdout,flush=True)
    if r.returncode: raise RuntimeError(f'{args[0]} returned {r.returncode}')
    return r.stdout


def save():
    state['updated_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(state_path)


def update_board_config():
    p=Path.home()/'lingbot-board/cfg.json';c=json.loads(p.read_text())
    c['pools']=[x for x in c.get('pools',[]) if not x['name'].startswith('lb_yolo_')]+[
        {'name':'lb_yolo_work_1111','label':'YOLO / 22k+8k 目标提示：采集、检测、BF16 评测',
         'dir':str(R/'pool'),'phases':str(R/'phases.txt')}]
    c['groups']=[x for x in c.get('groups',[]) if not x['name'].startswith('lb_yolo_')]+[
        {'name':f'lb_yolo_{m}_1111','label':f'22k+8k → {label}（2000 步，多卡接力）',
         'dir':str(R/'runs'/m),'log_glob':str(R/'logs'/f'lb_yolo_{m}_1111_*.log'),
         'step_regex':r'Step: (\d+)/(\d+) \[','ckpt_glob':str(R/'runs'/m/'checkpoints/global_step_*')}
        for m,label in [('text','纯文本'),('box','画框混训'),('coords','坐标混训')]]
    tmp=p.with_suffix('.yolo.tmp');tmp.write_text(json.dumps(c,ensure_ascii=False,indent=2));tmp.replace(p)


last_board=0
update_board_config()
while time.time()<deadline:
    try:
        episodes=list((R/'episodes').glob('*/complete.json'))
        records=[json.loads(p.read_text()) for p in episodes]
        state['episodes_complete']=len(records)
        smoke=[json.loads((R/'episodes'/f'{i:04d}'/'complete.json').read_text())
               for i in range(6) if (R/'episodes'/f'{i:04d}'/'complete.json').exists()]
        if len(smoke)==6 and all(x.get('metric_version')==2 for x in smoke) and not (R/'SMOKE_OK').exists():
            if not all(x.get('scene_version')==2 and x.get('expert_first_approach_correct') for x in smoke):
                state['blocked']='Expert first-approach calibration failed; inspect smoke trajectories.'
                save();break
            (R/'SMOKE_OK').write_text('six expert combinations passed\n')
            state.pop('blocked',None)
        if (R/'SMOKE_OK').exists() and not state.get('pool_id'):
            # Recover a submission if the SSH/logging process died after sbatch accepted it.
            existing=command(['squeue','-h','-u',os.environ['USER'],'-n',NAMES[0],'-o','%A']).split()
            if existing: state['pool_id']=existing[0]
            else:
                output=command(['bash',str(KIT/'pool/submit_pool.sh'),str(R/'configs/pool.sh')])
                state['pool_id']=output.split('array ')[1].split()[0]
            save()
        if len(records)==120 and all(x.get('scene_version')==2 for x in records):
            (R/'COLLECTION_READY').touch()
        for variant in ('text','box','coords'):
            ready=all((R/'lerobot'/v/'READY').exists() for v in ('text',variant))
            if ready and not state.get(f'{variant}_submitted'):
                registry=R/'runs'/variant/'.handoff/registry.tsv'
                submitted={line.split('\t')[2] for line in registry.read_text().splitlines()} if registry.exists() else set()
                # List members by sourcing the actual current kit configuration.
                config=R/'configs'/f'group_{variant}.sh'
                names=command(['bash','-c','source "$1"; for x in "${MEMBERS[@]}"; do printf "%s\\n" "${x%%|*}"; done','bash',str(config)]).splitlines()
                missing=[n for n in names if n not in submitted]
                if missing:
                    env=dict(os.environ,ONLY=' '.join(missing))
                    command(['bash',str(KIT/'train/submit_group.sh'),str(config)],timeout=900,env=env)
                state[f'{variant}_submitted']=True;save()
            if (R/'runs'/variant/'DONE').exists():
                if (R/'YOLO_READY').exists(): (R/f'{variant}_YOLO_EVAL_READY').touch()
                if not state.get(f'{variant}_queue_cleaned'):
                    # Cancel only pending backup members of this completed experiment group.
                    ids=command(['squeue','-h','-u',os.environ['USER'],'-n',f'lb_yolo_{variant}_1111','-t','PD','-o','%i']).split()
                    if ids: command(['scancel',*ids])
                    state[f'{variant}_queue_cleaned']=True;save()
        command(['python3',str(R/'code/summarize.py')],timeout=30)
        state['results_complete']=len(list((R/'results').glob('*/*/result.json')))
        if time.time()-last_board>120:
            # Existing authenticated private dashboard pipeline; no access changes.
            try: command(['flock','-n',str(Path.home()/'lingbot-board/.lock'),'env','FORCE=1','bash',str(Path.home()/'lingbot-board/runner.sh')],timeout=300)
            except Exception as exc: state['board_error']=str(exc)
            last_board=time.time()
        state.pop('last_error',None);save()
        if state['results_complete']==510 and (R/'YOLO_READY').exists():
            state['complete']=True;save();break
    except Exception as exc:
        state['last_error']=repr(exc);save();print(repr(exc),flush=True)
    time.sleep(60)
else:
    state['deadline_reached']=True;save()
    # Bounded run: stop only these experiment jobs at the configured pre-departure deadline.
    for name in NAMES:
        try:
            ids=command(['squeue','-h','-u',os.environ['USER'],'-n',name,'-o','%i']).split()
            if ids: command(['scancel',*ids])
        except Exception as exc: print(repr(exc),flush=True)
