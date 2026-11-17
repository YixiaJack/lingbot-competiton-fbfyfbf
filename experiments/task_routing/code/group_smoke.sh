# kit group for the routing smoke test: one big-memory GPU, first member to start runs smoke.sh, the rest cancel
source $SCRATCH/lingbot/tmp/gpu_queue_kit/config.example.sh
LOG_DIR=$SCRATCH/lingbot/final_route_20261112/logs
GROUP_NAME=lb_route_smoke_1112
GROUP_DIR=$SCRATCH/lingbot/final_route_20261112/runs_smoke
TRAIN_CMD='bash $SCRATCH/lingbot/LingBotRoboTwin/route/smoke.sh'
MEMBERS=(
  "h200x1_dept|15|${SLURM_ACCOUNT_ADVANCED:-<slurm-account-advanced>}|0|--gres=gpu:h200:1 --constraint=h200 --cpus-per-task=16 --mem=200G --time=04:00:00|MICRO_BS=2"
  "b200x1_dept_prem|5|${SLURM_ACCOUNT_ADVANCED:-<slurm-account-advanced>}|1|--gres=gpu:b200:1 --constraint=b200 --cpus-per-task=16 --mem=200G --time=04:00:00|MICRO_BS=2"
  "b200x1|3|${SLURM_ACCOUNT_GENERAL:-<slurm-account-general>}|1|--gres=gpu:b200:1 --constraint=b200 --cpus-per-task=16 --mem=200G --time=04:00:00|MICRO_BS=2"
  "h200x1|3|${SLURM_ACCOUNT_GENERAL:-<slurm-account-general>}|1|--gres=gpu:h200:1 --constraint=h200 --cpus-per-task=16 --mem=200G --time=04:00:00|MICRO_BS=2"
)
