#!/usr/bin/env bash
# Convert the 50 aloha-agilex_clean_50 tasks to LeRobot datasets, one array element per task.
#   scripts/hpc/submit.sh --cpu scripts/hpc/convert.sbatch            (all 50)
#   scripts/hpc/submit.sh --cpu scripts/hpc/convert.sbatch --array=3  (one task, index into configs/tasks50.json)
# Skips tasks whose dataset already finalized (meta/info.json present); safe to requeue.
#SBATCH --job-name=lb_convert
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/lingbot/logs/convert_%A_%a.log
set -euo pipefail
cd "${WORK_DIR:-/scratch/${USER}/lingbot/LingBotRoboTwin}"
source scripts/hpc/env.sh
TASK=$(python3 -c "import json;print(json.load(open('configs/tasks50.json'))[${SLURM_ARRAY_TASK_ID}])")
OUT="${DATA_DIR}/lerobot/robotwin_clean/${TASK}-aloha-agilex_clean_50-50"
if [[ -f "${OUT}/.done" ]]; then echo "already converted: ${OUT}"; exit 0; fi
echo "=== ${TASK} on $(hostname) $(date)"
scripts/hpc/run_in_env.sh python src/convert_robotwin_to_lerobot.py \
  --raw_dir "${DATA_DIR}/raw/${TASK}/aloha-agilex_clean_50" --out_dir "${OUT}"
touch "${OUT}/.done"
echo "=== done $(date)"
