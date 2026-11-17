#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE=$SCRATCH/lingbot
export WORK_DIR="$BASE/LingBotRoboTwin"
source "$WORK_DIR/scripts/hpc/env.sh"
export PYTHONPATH="$HERE:$ROBOTWIN_DIR:${PYTHONPATH:-}"
export VK_ICD_FILENAMES="$WORK_DIR/configs/nvidia_icd.json"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export YOLO_CONFIG_DIR=$SCRATCH/lingbot/final_yolo_20261111/yolo_settings
mode="$1"; shift
cd "$ROBOTWIN_DIR"
if [[ "$mode" == sim ]]; then
  exec bash "$WORK_DIR/scripts/hpc/run_in_env.sh" bash -c 'source "$PROJ/miniforge3/etc/profile.d/conda.sh"; conda activate RoboTwin; exec "$@"' bash "$@"
else
  exec bash "$WORK_DIR/scripts/hpc/run_in_env.sh" "$@"
fi
