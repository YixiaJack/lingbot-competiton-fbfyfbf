#!/usr/bin/env bash
# Normalization statistics over the 50 clean datasets only (upstream assets/norm_stats/robotwin.json was
# computed on clean + randomized data, which this competition forbids for training).
#   scripts/hpc/submit.sh scripts/hpc/norm_stats.sbatch   (GPU: lingbotvla queries the device at import)
#SBATCH --job-name=lb_norm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|h100|h200
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/lingbot/logs/norm_stats_%j.log
set -euo pipefail
cd "${WORK_DIR:-/scratch/${USER}/lingbot/LingBotRoboTwin}"
source scripts/hpc/env.sh
OUT="${WORK_DIR}/configs/norm_stats/robotwin_clean.json"
mkdir -p "$(dirname "${OUT}")"
# the robot config's norm_stats path must exist when the dataset is built; stats are not applied here
# (do_nomalize=False). Seed it with upstream's file, then overwrite it with the clean-only result.
[[ -f "${OUT}" ]] || cp "${LINGBOT_DIR}/assets/norm_stats/robotwin.json" "${OUT}"
rm -f "${OUT}.computed"
cd "${LINGBOT_DIR}"
"${WORK_DIR}/scripts/hpc/run_in_env.sh" torchrun --nnodes=1 --nproc-per-node=1 --master-port=$(( 20000 + RANDOM % 20000 )) scripts/compute_norm_stats.py ./configs/vla/norm_compute/post_data.yaml \
  --data.robot_name robotwin_clean \
  --data.train_path "${WORK_DIR}/configs/robotwin_clean.txt" \
  --data.robot_config_root "${WORK_DIR}/configs/robot_configs" \
  --data.norm_path "${OUT}" \
  --data.num_workers 16
cmp -s "${OUT}" "${LINGBOT_DIR}/assets/norm_stats/robotwin.json" && { echo "norm stats were not rewritten"; exit 1; }
touch "${OUT}.computed"
echo "=== norm stats written: ${OUT}"
