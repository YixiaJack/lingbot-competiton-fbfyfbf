#!/usr/bin/env bash
# Lists, phases and the two pool configs (general preemptible + non-preemptible regular) for the detector pipeline. Submits nothing.
set -euo pipefail
R=$SCRATCH/lingbot/final_real_20261112; BASE=$SCRATCH/lingbot; C=$R/code
DEPS="PYTHONPATH=$BASE/final_yolo_20261111/deps"
mkdir -p $R/lists $R/configs $R/pool $R/det
seq 0 109 > $R/lists/det_items.txt
echo 0 > $R/lists/single.txt
cat > $R/phases.txt <<P
det|$R/lists/det_items.txt|bash $C/runtime.sh sim python $C/det_render.py --root $R --item {item}|$R/assets/assets.json
yolo|$R/lists/single.txt|$DEPS bash $C/runtime.sh model python $C/train_yolo_real.py --root $R|$R/det/RENDER_READY
select|$R/lists/single.txt|$DEPS bash $C/runtime.sh model python $C/select_eval.py --root $R|$R/det/YOLO_READY
P
common="ROOT=$R
LOG_DIR=$R/logs
PREEMPT_COMMENT=\"preemption=yes;preemption_partitions_only=yes;requeue=true\"
POOL_NAME=lb_real_work_1112
POOL_DIR=$R/pool
POOL_PHASES=$R/phases.txt
POOL_SLOTS=1
POOL_WAIT_S=60"
printf '%s\n' "$common" 'POOL_ACCOUNT=${SLURM_ACCOUNT_GENERAL:-<slurm-account-general>}' 'POOL_PREEMPT=1' 'POOL_ARRAY="0-9"' \
  'POOL_SBATCH="--gres=gpu:1 --constraint=a100|l40s|h100|h200 --cpus-per-task=8 --mem=64G --time=12:00:00"' > $R/configs/pool_general.sh
# non-preemptible regular partitions (not preemptible, owner priority), same accounts as the kit's dept members
printf '%s\n' "$common" 'POOL_ACCOUNT=${SLURM_ACCOUNT_PRIORITY:-<slurm-account-priority>}' 'POOL_PREEMPT=0' 'POOL_ARRAY="0-1"' \
  'POOL_SBATCH="--gres=gpu:h100:1 --constraint=h100 --cpus-per-task=8 --mem=64G --time=12:00:00"' > $R/configs/pool_dept_h100.sh
printf '%s\n' "$common" 'POOL_ACCOUNT=${SLURM_ACCOUNT_ADVANCED:-<slurm-account-advanced>}' 'POOL_PREEMPT=0' 'POOL_ARRAY="0"' \
  'POOL_SBATCH="--gres=gpu:h200:1 --constraint=h200 --cpus-per-task=8 --mem=64G --time=12:00:00"' > $R/configs/pool_dept_h200.sh
echo ok
