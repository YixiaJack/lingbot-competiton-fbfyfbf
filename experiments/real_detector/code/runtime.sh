#!/usr/bin/env bash
# runtime.sh sim|model <command...>: RoboTwin simulator env or LingBot model env, inside the shared Apptainer overlay.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
BASE=$SCRATCH/lingbot
export WORK_DIR="$BASE/LingBotRoboTwin"
source "$WORK_DIR/scripts/hpc/env.sh"
export PYTHONPATH="$HERE:$ROBOTWIN_DIR:${PYTHONPATH:-}"
export VK_ICD_FILENAMES="$WORK_DIR/configs/nvidia_icd.json"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export YOLO_CONFIG_DIR="$ROOT/yolo_settings"
mode="$1"; shift
cd "$ROBOTWIN_DIR"
if [[ "$mode" == sim ]]; then
  exec bash "$WORK_DIR/scripts/hpc/run_in_env.sh" bash -c 'source "$PROJ/miniforge3/etc/profile.d/conda.sh"; conda activate RoboTwin; exec "$@"' bash "$@"
else
  exec bash "$WORK_DIR/scripts/hpc/run_in_env.sh" "$@"
fi
