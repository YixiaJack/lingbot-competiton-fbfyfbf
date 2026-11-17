#!/usr/bin/env bash
# One-command training on a single multi-GPU machine (no Slurm). Same command as scripts/hpc/train.sbatch.
#   LINGBOT_DIR=/path/to/lingbot-vla-v2 CKPT_DIR=/path/to/ckpts DATA_DIR=/path/to/data/lerobot/robotwin_clean \
#   OUTPUT_DIR=/path/to/runs/main_b200x4 bash train.sh      # reproduces the submitted 40k checkpoint
# Paths inside configs/*.yaml / configs/robotwin_clean.txt point at our cluster ($SCRATCH/lingbot/...);
# this script rewrites them to the variables above in a temporary copy.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LINGBOT_DIR:?}" "${CKPT_DIR:?}" "${DATA_DIR:?}" "${OUTPUT_DIR:?}"
# defaults = the submitted run (main_b200x4, global_step_40000): global batch 128, 40k steps. With other GPU
# counts the global batch stays 128 (gradient accumulation); on 80 GB cards use MICRO_BS=16.
MICRO_BS="${MICRO_BS:-32}"; MAX_STEPS="${MAX_STEPS:-40000}"; SAVE_STEPS="${SAVE_STEPS:-2500}"; GLOBAL_BS="${GLOBAL_BS:-128}"
NGPU=$(nvidia-smi -L | wc -l)
CFG="${OUTPUT_DIR}/_cfg"; mkdir -p "${CFG}/robot_configs" "${CFG}/norm_stats"
OLD='\$SCRATCH/lingbot'
sub() { sed -e "s#${OLD}/ckpts#${CKPT_DIR}#g" -e "s#${OLD}/data/lerobot/robotwin_clean#${DATA_DIR}#g" \
            -e "s#${OLD}/LingBotRoboTwin/configs#${CFG}#g" -e "s#${OLD}/runs/robotwin_clean#${OUTPUT_DIR}#g" "$1"; }
sub "${HERE}/configs/robotwin_clean.yaml" > "${CFG}/robotwin_clean.yaml"
sub "${HERE}/configs/robotwin_clean.txt" > "${CFG}/robotwin_clean.txt"
sub "${HERE}/configs/robot_configs/robotwin_clean.yaml" > "${CFG}/robot_configs/robotwin_clean.yaml"
cp "${HERE}/configs/norm_stats/robotwin_clean.json" "${CFG}/norm_stats/"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${LINGBOT_DIR}"
torchrun --nnodes=1 --nproc-per-node "${NGPU}" --master-port="${MASTER_PORT:-29531}" \
  tasks/vla/train_lingbotvla.py "${CFG}/robotwin_clean.yaml" \
  --data.norm_stats_file "${CFG}/norm_stats/robotwin_clean.json" \
  --train.output_dir "${OUTPUT_DIR}" \
  --train.micro_batch_size "${MICRO_BS}" --train.global_batch_size "${GLOBAL_BS}" \
  --train.max_steps "${MAX_STEPS}" --train.save_steps "${SAVE_STEPS}" ${TRAIN_ARGS:-}
