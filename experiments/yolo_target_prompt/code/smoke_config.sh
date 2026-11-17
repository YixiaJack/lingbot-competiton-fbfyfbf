ROOT=$SCRATCH/lingbot/final_yolo_20261111
LOG_DIR="$ROOT/logs"
PREEMPT_COMMENT="preemption=yes;preemption_partitions_only=yes;requeue=true"
POOL_NAME="lb_yolo_smoke_1111"
POOL_DIR="$ROOT/pool_smoke"
POOL_PHASES="$ROOT/smoke_phases.txt"
POOL_ACCOUNT="${SLURM_ACCOUNT_ADVANCED:-<slurm-account-advanced>}"
POOL_PREEMPT=1
POOL_ARRAY="0"
POOL_SBATCH="--gres=gpu:h200:1 --constraint=h200 --cpus-per-task=8 --mem=64G --time=02:00:00"
POOL_SLOTS=1
POOL_WAIT_S=30
