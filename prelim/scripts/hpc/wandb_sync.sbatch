#!/usr/bin/env bash
# CPU sidecar: mirror a training run's TensorBoard scalars to wandb while the training job is in the queue.
#   RUN_NAME=main_b200x4 TRAIN_JOB=<train job id> scripts/hpc/submit.sh --cpu scripts/hpc/wandb_sync.sbatch
#   RUN_NAME=<run> TRAIN_JOB_NAME=lb_<run> ...   (follow a handoff group by job name instead of one job id)
#   TB_ARGS="--since <unix time> --id_suffix r2"  (run restarted from step 0: mirror only the new attempt, new wandb run)
# wandb credentials come from ~/.netrc (wandb login); the training config keeps use_wandb: false.
#SBATCH --job-name=lb_wandb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/%u/lingbot/logs/wandb_%x_%j.log
set -euo pipefail
cd "${WORK_DIR:-/scratch/${USER}/lingbot/LingBotRoboTwin}"
source scripts/hpc/env.sh
: "${RUN_NAME:?}"
[[ -n "${TRAIN_JOB:-}${TRAIN_JOB_NAME:-}" ]] || { echo "TRAIN_JOB or TRAIN_JOB_NAME required"; exit 1; }
STOP="${TMPDIR}/wandb_stop_${SLURM_JOB_ID}"; rm -f "${STOP}"
ENV_NAME=lingbotvla scripts/hpc/run_in_env.sh python -u src/tb_to_wandb.py \
  --run_dir "${RUN_DIR}/${RUN_NAME}" --project "${WANDB_PROJECT:-lingbot-robotwin}" --stop_file "${STOP}" ${TB_ARGS:-} &
PID=$!
# keep mirroring while the training job (TRAIN_JOB) or any job named TRAIN_JOB_NAME (handoff group) is queued
training_alive() {
  if [[ -n "${TRAIN_JOB_NAME:-}" ]]; then [[ -n "$(squeue -h -u "${USER}" -n "${TRAIN_JOB_NAME}" 2>/dev/null)" ]]
  else [[ -n "$(squeue -h -j "${TRAIN_JOB}" 2>/dev/null)" ]]; fi
}
while kill -0 ${PID} 2>/dev/null && training_alive; do sleep 120; done
touch "${STOP}"
wait ${PID} || true
echo "=== wandb sync end $(date)"
