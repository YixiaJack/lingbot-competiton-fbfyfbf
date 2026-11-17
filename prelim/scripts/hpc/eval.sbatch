#!/usr/bin/env bash
# Official RoboTwin evaluation, one array element per task (100 episodes each, seed 0, FP32 inference).
#   CKPT=.../hf_ckpt EVAL_NAME=main_20k TASK_CONFIG=demo_clean      scripts/hpc/submit.sh scripts/hpc/eval.sbatch
#   CKPT=.../hf_ckpt EVAL_NAME=main_20k TASK_CONFIG=demo_randomized scripts/hpc/submit.sh scripts/hpc/eval.sbatch
# Finished tasks (<task>.result) are skipped, so a preempted/requeued element just reruns its own task.
# Smoke test: EVAL_TEST_NUM=3 ... --array=0
#SBATCH --job-name=lb_eval
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# GPU types: never h100|h200, those are kept for training
#SBATCH --constraint=l40s|a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/lingbot/logs/eval_%x_%A_%a.log
set -euo pipefail
cd "${WORK_DIR:-/scratch/${USER}/lingbot/LingBotRoboTwin}"
source scripts/hpc/env.sh
: "${CKPT:?}" "${EVAL_NAME:?}" "${TASK_CONFIG:?}"
TASK=$(python3 -c "import json;print(json.load(open('configs/tasks50.json'))[${SLURM_ARRAY_TASK_ID}])")
OUT="${PROJ}/eval/${EVAL_NAME}/${TASK_CONFIG}"
[[ -f "${OUT}/${TASK}.result" ]] && { echo "done already: $(cat "${OUT}/${TASK}.result")"; exit 0; }
echo "=== eval ${TASK} ${TASK_CONFIG} on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date)"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-${WORK_DIR}/configs/nvidia_icd.json}"
scripts/hpc/run_in_env.sh bash src/eval_one_task.sh "${CKPT}" "${TASK}" "${TASK_CONFIG}" "${OUT}"
echo "=== end $(date)"
