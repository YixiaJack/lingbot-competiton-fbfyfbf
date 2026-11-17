"""Generate immutable experiment settings and scheduler inputs; submits nothing."""
import os
import json
from pathlib import Path
import yaml

ROOT=Path(os.path.expandvars('$SCRATCH/lingbot/final_yolo_20261111'))
BASE=ROOT.parent
WORK=BASE/'LingBotRoboTwin'
KIT=BASE/'tmp/gpu_queue_kit'
INIT=BASE/'runs/strong_scratch22k_cont8k/checkpoints/global_step_8000/hf_ckpt'
ROOT.mkdir(exist_ok=True)
for name in ('configs','logs','lists','runs'):
    (ROOT/name).mkdir(exist_ok=True)
settings={'init':str(INIT),'init_source':'22k+8k explicitly requested; overrides the plan\'s official-base start',
          'source_repo':'<notes-repo>','source_plan_blob':'7d3a711e4c82f8325ae704d8e7d2b1194bd9b4ad',
          'transcript':'02-团队讨论_0108-0237.txt, 2:17:50–2:19:30',
          'train_episodes':120,'steps_each':2000,'global_batch':128,'lr':1e-5,
          'variants':{'text':'all plain','box':'50% plain + 50% static frame-zero boxes',
                      'coords':'50% plain + 50% frame-zero coordinate instructions'},
          'proxy':'3 colored blocks + 3 colored plates in RoboTwin; not actual final fruit/cosmetics objects',
          'heldout_combos':[['red','yellow'],['green','cyan'],['blue','magenta']],
          'evaluation_precision':'BF16','adoption_threshold_pp':15,'fallback_requirement':'no plain-input regression',
          'scope':'pre-event simulation only; no resulting checkpoint is a final-competition artifact',
          'metric':'first approach proxy: gripper TCP within 10cm, moved 4cm closer, z<0.9m; no approach incorrect',
          'budget_end_utc':'2026-11-13T10:00:00Z'}
(ROOT/'experiment.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2))
for variant in ('text','box','coords'):
    train_list=ROOT/'lists'/f'{variant}.txt'
    train_list.write_text(f'robotwin_clean {ROOT}/lerobot/text\n'+
                         (f'robotwin_clean {ROOT}/lerobot/{variant}\n' if variant!='text' else ''))
    cfg=yaml.safe_load((WORK/'configs/robotwin_clean.yaml').read_text())
    cfg['model']['model_path']=str(INIT)
    cfg['data'].update(train_path=str(train_list),image_augment=False,num_workers=4)
    cfg['train'].update(output_dir=str(ROOT/'runs'/variant),max_steps=2000,save_steps=500,
                        lr=1e-5,lr_min=5e-6,lr_warmup_ratio=.03,tokenizer_max_length=128,
                        global_batch_size=128,enable_resume=True)
    (ROOT/'configs'/f'{variant}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    # Reuse the current kit verbatim, including all six non-preemptible options.
    group=f'''source {KIT}/config.example.sh
LOG_DIR={ROOT}/logs
GROUP_NAME=lb_yolo_{variant}_1111
GROUP_DIR={ROOT}/runs/{variant}
TRAIN_CMD='bash {ROOT}/code/train.sh {variant}'
'''
    (ROOT/'configs'/f'group_{variant}.sh').write_text(group)

pool=f'''ROOT={ROOT}
LOG_DIR={ROOT}/logs
PREEMPT_COMMENT="preemption=yes;preemption_partitions_only=yes;requeue=true"
POOL_NAME=lb_yolo_work_1111
POOL_DIR={ROOT}/pool
POOL_PHASES={ROOT}/phases.txt
POOL_ACCOUNT=${{SLURM_ACCOUNT_GENERAL:-<slurm-account-general>}}
POOL_PREEMPT=1
POOL_ARRAY="0-7"
POOL_SBATCH="--gres=gpu:1 --constraint=a100|l40s|h100|h200 --cpus-per-task=8 --mem=64G --time=12:00:00"
POOL_SLOTS=1
POOL_WAIT_S=60
'''
(ROOT/'configs/pool.sh').write_text(pool)
(ROOT/'lists/collect.txt').write_text(''.join(f'{i}\n' for i in range(120)))
(ROOT/'lists/variants.txt').write_text('text\nbox\ncoords\n')
(ROOT/'lists/single.txt').write_text('0\n')
(ROOT/'lists/main_eval.txt').write_text(''.join(f'{i}\n' for i in range(60)))
(ROOT/'lists/heldout_eval.txt').write_text(''.join(f'{i}\n' for i in range(30,60)))
phases=[]
for run,mode,source,items,gate in (
        [('base',m,'oracle','heldout_eval','SMOKE_OK') for m in ('text','box','coords')]+
        [(m,m,'oracle','main_eval',f'runs/{m}/DONE') for m in ('text','box','coords')]+
        [(m,c,s,'heldout_eval',f'runs/{m}/DONE') for m in ('box','coords')
         for c,s in [('text','oracle'),(m,'jitter'),(m,'wrong10')]]):
    phases.append(f'{run}_{mode}_{source}|{ROOT}/lists/{items}.txt|bash {ROOT}/code/eval_item.sh {run} {mode} {source} {{item}}|{ROOT}/{gate}')
# Preparation precedes evaluation so workers don't delay the critical training path.
prep=[f'collect|{ROOT}/lists/collect.txt|bash {ROOT}/code/runtime.sh sim python {ROOT}/code/collect.py --root {ROOT} --item {{item}}|{ROOT}/SMOKE_OK',
      f'convert|{ROOT}/lists/variants.txt|bash {ROOT}/code/runtime.sh model python {ROOT}/code/prepare_data.py --root {ROOT} --variant {{item}}|{ROOT}/COLLECTION_READY',
      f'yolo|{ROOT}/lists/single.txt|PYTHONPATH={ROOT}/deps bash {ROOT}/code/runtime.sh model python {ROOT}/code/train_yolo.py --root {ROOT}|{ROOT}/COLLECTION_READY']
for m in ('box','coords'):
    phases.append(f'{m}_{m}_yolo|{ROOT}/lists/heldout_eval.txt|bash {ROOT}/code/eval_item.sh {m} {m} yolo {{item}}|{ROOT}/{m}_YOLO_EVAL_READY')
(ROOT/'phases.txt').write_text('\n'.join(prep+phases)+'\n')
print(json.dumps(settings,ensure_ascii=False,indent=2))
