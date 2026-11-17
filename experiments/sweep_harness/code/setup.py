"""Lists, pool phases, pool configs, training config and training group for the sweeping-harness experiment.
Submits nothing. Re-run after editing; it only rewrites config files.
"""
import os
from pathlib import Path
import yaml

R = Path(os.path.expandvars('$SCRATCH/lingbot/sweep_harness_20261114')); BASE = R.parent; C = R / 'code'
KIT = BASE / 'tmp/gpu_queue_kit'; WORK = BASE / 'LingBotRoboTwin'
S22 = BASE / 'runs/strong_scratch22k_cont8k/checkpoints/global_step_8000/hf_ckpt'   # prelim RoboTwin model (LingBot-VLA 2.0)
N_COLLECT, N_VAL, N_EVAL = 190, 10, 30
RUN, STEPS = 'sweep_s22', 5000
DEPS_LEROBOT = f'PYTHONPATH={BASE}/final_yolo_20261111/deps'

for d in ('configs', 'lists', 'runs', 'logs', 'pool', 'episodes', 'eval_seeds', 'results', 'judge'):
    (R / d).mkdir(exist_ok=True)
VAL = list(range(N_COLLECT - N_VAL, N_COLLECT))   # held out for the judgement test: collected first so it can start early
(R / 'lists/collect.txt').write_text(''.join(f'{i}\n' for i in VAL + [i for i in range(N_COLLECT) if i not in VAL]))
(R / 'lists/val_items.txt').write_text(''.join(f'{i}\n' for i in range(N_COLLECT - N_VAL, N_COLLECT)))
(R / 'lists/eval_items.txt').write_text(''.join(f'{i}\n' for i in range(N_EVAL)))
(R / 'lists/single.txt').write_text('0\n')
SHARDS = [f'{v}_{k}' for v in ('skills', 'full') for k in range(8)]   # prepare_data.NSHARDS
(R / 'lists/variants.txt').write_text(''.join(f'{x}\n' for x in SHARDS))
sim = f'bash {C}/runtime.sh sim python'
(R / 'lists/judge.txt').write_text('v1\nv2\n')
(R / 'phases.txt').write_text(
    f'eval|{R}/lists/eval_chunks.txt|bash {C}/eval_batch.sh {RUN} {{item}}|{R}/TRAIN_DONE\n'   # no '|' inside a command: the pool splits lines on it
    f'judge|{R}/lists/judge.txt|{sim} {C}/judge.py --root {R} --version {{item}}|{R}/VAL_READY\n'
    f'prep|{R}/lists/variants.txt|{DEPS_LEROBOT} bash {C}/runtime.sh model python {C}/prepare_data.py --root {R} --variant {{item}}|{R}/COLLECTION_READY\n'
    f'collect|{R}/lists/collect.txt|{sim} {C}/collect.py --root {R} --item {{item}}|{R}/EXPERT_OK\n'
    f'seeds|{R}/lists/eval_items.txt|{sim} {C}/rollout.py --root {R} --item {{item}} --seed-only|{R}/EXPERT_OK\n')
common = f"""ROOT={R}
LOG_DIR={R}/logs
PREEMPT_COMMENT="preemption=yes;preemption_partitions_only=yes;requeue=true"
POOL_NAME=lb_sweep_work
POOL_DIR={R}/pool
POOL_PHASES={R}/phases.txt
POOL_SLOTS=1
POOL_WAIT_S=120
"""
(R / 'configs/pool_general.sh').write_text(common + 'POOL_ACCOUNT=${SLURM_ACCOUNT_GENERAL:-<slurm-account-general>}\nPOOL_PREEMPT=1\nPOOL_ARRAY="0-11"\n'
    'POOL_SBATCH="--gres=gpu:1 --constraint=a100|l40s|h100|h200 --cpus-per-task=8 --mem=64G --time=08:00:00"\n')
(R / 'configs/pool_dept.sh').write_text(common + 'POOL_ACCOUNT=${SLURM_ACCOUNT_PRIORITY:-<slurm-account-priority>}\nPOOL_PREEMPT=0\nPOOL_ARRAY="0-1"\n'
    'POOL_SBATCH="--gres=gpu:h100:1 --constraint=h100 --cpus-per-task=8 --mem=64G --time=08:00:00"\n')

# training: one model on both datasets (skill segments + full episodes), Qwen3-VL frozen, action expert trained
lst = R / 'lists/train.txt'
lst.write_text(''.join(f'robotwin_clean {R}/lerobot/{x}\n' for x in SHARDS))
cfg = yaml.safe_load((WORK / 'configs/robotwin_clean.yaml').read_text())
cfg['model']['model_path'] = str(S22)
cfg['data'].update(train_path=str(lst), image_augment=False, num_workers=4)
cfg['train'].update(output_dir=str(R / 'runs' / RUN), max_steps=STEPS, save_steps=1000, lr=2e-5, lr_min=2e-6,
                    lr_warmup_ratio=.03, tokenizer_max_length=128, global_batch_size=128, enable_resume=True,
                    train_expert_only=True, freeze_vision_encoder=True)
(R / 'configs' / f'{RUN}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
(R / 'configs' / f'group_{RUN}.sh').write_text(
    f"source {KIT}/config.example.sh\nLOG_DIR={R}/logs\nGROUP_NAME=lb_sweep_train\nGROUP_DIR={R}/runs/{RUN}\n"
    f"TRAIN_CMD='bash {C}/train.sh {RUN} {STEPS}'\n")
print('ok')
