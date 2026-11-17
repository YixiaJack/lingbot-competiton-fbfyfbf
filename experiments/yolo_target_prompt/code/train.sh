#!/usr/bin/env bash
set -euo pipefail
ROOT=$SCRATCH/lingbot/final_yolo_20261111
VARIANT="$1"
export LINGBOT_DIR=$SCRATCH/lingbot/dev/lingbot-vla-v2-dino
OUT="$ROOT/runs/$VARIANT"
[[ -f "$OUT/DONE" ]] && exit 0
[[ -f "$ROOT/lerobot/text/READY" && -f "$ROOT/lerobot/$VARIANT/READY" ]] || exit 2
GLOBAL=128
MICRO="${MICRO_BS:-4}"
ACC=$(( GLOBAL / (MICRO * NGPU) ))
(( ACC * MICRO * NGPU == GLOBAL )) || exit 2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LINGBOT_TASK_SAMPLING=uniform LINGBOT_SEEN_POOL=0 LINGBOT_FRAME_WEIGHTS=0
export LINGBOT_TIME_SAVE_MIN=15 LINGBOT_VISION_LR_SCALE=0.1
export PYTHONPATH="$LINGBOT_DIR:$SCRATCH/lingbot/LingBotRoboTwin/src"
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])')
BOOL_GC=false; [[ "${GRAD_CKPT:-1}" == 1 ]] && BOOL_GC=true
BOOL_FS=false; [[ "${FULL_SHARD:-1}" == 1 ]] && BOOL_FS=true
cd "$LINGBOT_DIR"
bash $SCRATCH/lingbot/LingBotRoboTwin/scripts/hpc/run_in_env.sh \
  torchrun --nnodes=1 --nproc-per-node "$NGPU" --master-port "$PORT" tasks/vla/train_lingbotvla.py \
  "$ROOT/configs/$VARIANT.yaml" --train.micro_batch_size "$MICRO" \
  --data.norm_stats_file $SCRATCH/lingbot/LingBotRoboTwin/configs/norm_stats/robotwin_clean.json \
  --train.gradient_accumulation_steps "$ACC" --train.global_batch_size "$GLOBAL" \
  --train.enable_gradient_checkpointing "$BOOL_GC" --train.enable_full_shard "$BOOL_FS"
[[ -f "$OUT/checkpoints/global_step_2000/hf_ckpt/model.safetensors.index.json" ]] || exit 3
touch "$OUT/checkpoints/global_step_2000/keep" "$OUT/DONE"
