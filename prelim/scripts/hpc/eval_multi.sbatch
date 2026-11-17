#!/usr/bin/env bash
# Official RoboTwin evaluation with several tasks sharing ONE GPU (each task: its own inference server + client,
# exactly as eval.sbatch / src/eval_one_task.sh). One eval task uses ~31 GB of GPU memory and is CPU-bound
# (simulation), so an 80 GB A100/H100 fits 2 tasks and a 141 GB H200 fits 4; L40S (46 GB) only fits 1 -> eval.sbatch.
#   CKPT=.../hf_ckpt EVAL_NAME=main40k TASK_CONFIG=demo_clean TASKS_PER_GPU=2 \
#     scripts/hpc/submit.sh scripts/hpc/eval_multi.sbatch --array=0-24 --constraint=a100 --cpus-per-task=16 --mem=128G
# Array element i runs tasks i*N .. i*N+N-1 of configs/tasks50.json (N = TASKS_PER_GPU, so --array=0-$((50/N - 1))).
# Finished tasks (<task>.result) are skipped and unfinished ones resume per episode (EVAL_STATE_FILE), so a
# preempted + requeued element only redoes the episode it was in.
#SBATCH --job-name=lb_eval2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# a100 by default; pass --constraint=h200 (TASKS_PER_GPU=4) explicitly once training no longer needs H200s
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/lingbot/logs/eval_%x_%A_%a.log
set -uo pipefail
cd "${WORK_DIR:-/scratch/${USER}/lingbot/LingBotRoboTwin}"
source scripts/hpc/env.sh
: "${CKPT:?}" "${EVAL_NAME:?}" "${TASK_CONFIG:?}"
N="${TASKS_PER_GPU:-2}"
OUT="${PROJ}/eval/${EVAL_NAME}/${TASK_CONFIG}"
mapfile -t TASKS < <(python3 -c "
import json; t = json.load(open('configs/tasks50.json')); i, n = ${SLURM_ARRAY_TASK_ID}, ${N}
print('\n'.join(t[i * n:(i + 1) * n]))")
echo "=== eval ${TASKS[*]} ${TASK_CONFIG} on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date)"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-${WORK_DIR}/configs/nvidia_icd.json}"
PIDS=()
for TASK in "${TASKS[@]}"; do
  if [[ -f "${OUT}/${TASK}.result" ]]; then echo "done already: ${TASK} $(cat "${OUT}/${TASK}.result")"; continue; fi
  scripts/hpc/run_in_env.sh bash src/eval_one_task.sh "${CKPT}" "${TASK}" "${TASK_CONFIG}" "${OUT}" \
    > "${LOG_DIR}/eval_${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TASK}.log" 2>&1 &
  PIDS+=($!)
  sleep 60  # stagger model loading / torch.compile warm-up
done
RC=0
for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
for TASK in "${TASKS[@]}"; do echo "${TASK}: $(cat "${OUT}/${TASK}.result" 2>/dev/null || echo unfinished)"; done
echo "=== end $(date)"
exit ${RC}
