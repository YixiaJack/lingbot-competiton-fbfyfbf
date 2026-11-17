#!/usr/bin/env bash
# Build the overlay on a GPU node (upstream env script checks torch.cuda.is_available()).
#   scripts/hpc/submit.sh scripts/hpc/build_overlay.sbatch
#SBATCH --job-name=lb_build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|h100|h200
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/lingbot/logs/build_overlay_%j.log
set -euo pipefail
cd "${WORK_DIR:-/scratch/${USER}/lingbot/LingBotRoboTwin}"
bash scripts/hpc/build_overlay.sh
