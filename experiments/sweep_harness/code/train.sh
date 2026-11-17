#!/usr/bin/env bash
# One training run (called by the kit's member.sbatch, which sets NGPU, MICRO_BS, GRAD_CKPT, FULL_SHARD).
# Same launcher as the real-scene A/B (train_real.sh); config from setup.py: Qwen3-VL frozen, action expert trained.
set -euo pipefail
R=$SCRATCH/lingbot/sweep_harness_20261114
NAME="$1"; STEPS="$2"
export LINGBOT_DIR=$SCRATCH/lingbot/dev/lingbot-vla-v2-dino
OUT="$R/runs/$NAME"
[[ -f "$OUT/DONE" ]] && exit 0
for v in $(cat $R/lists/variants.txt); do [[ -f "$R/lerobot/$v/READY" ]] || exit 2; done
GLOBAL=128; MICRO="${MICRO_BS:-4}"; ACC=$(( GLOBAL / (MICRO * NGPU) ))
(( ACC * MICRO * NGPU == GLOBAL )) || { echo "global $GLOBAL not divisible by micro $MICRO x $NGPU GPUs"; exit 2; }
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LINGBOT_TASK_SAMPLING=uniform LINGBOT_SEEN_POOL=0 LINGBOT_FRAME_WEIGHTS=0
export LINGBOT_TIME_SAVE_MIN=15 LINGBOT_VISION_LR_SCALE=0.1
export PYTHONPATH="$LINGBOT_DIR:$SCRATCH/lingbot/LingBotRoboTwin/src"
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])')
GC=false; [[ "${GRAD_CKPT:-1}" == 1 ]] && GC=true
FS=false; [[ "${FULL_SHARD:-1}" == 1 ]] && FS=true
cd "$LINGBOT_DIR"
bash $SCRATCH/lingbot/LingBotRoboTwin/scripts/hpc/run_in_env.sh \
  torchrun --nnodes=1 --nproc-per-node "$NGPU" --master-port "$PORT" tasks/vla/train_lingbotvla.py \
  "$R/configs/$NAME.yaml" --train.micro_batch_size "$MICRO" \
  --data.norm_stats_file $SCRATCH/lingbot/LingBotRoboTwin/configs/norm_stats/robotwin_clean.json \
  --train.gradient_accumulation_steps "$ACC" --train.global_batch_size "$GLOBAL" \
  --train.enable_gradient_checkpointing "$GC" --train.enable_full_shard "$FS"
[[ -f "$OUT/checkpoints/global_step_$STEPS/hf_ckpt/model.safetensors.index.json" ]] || exit 3
touch "$OUT/checkpoints/global_step_$STEPS/keep" "$OUT/DONE"
