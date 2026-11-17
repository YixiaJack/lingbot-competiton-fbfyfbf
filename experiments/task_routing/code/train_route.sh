#!/usr/bin/env bash
# One arm of the routing experiment. Called by the kit's member.sbatch (sets NGPU, MICRO_BS, GRAD_CKPT,
# FULL_SHARD) or directly by smoke.sh.   train_route.sh ARM STEPS [SAVE_STEPS] [--smoke]
set -euo pipefail
R2=$SCRATCH/lingbot/final_route_20261112
ARM="$1"; STEPS="$2"; SAVE="${3:-1500}"; SMOKE="${4:-}"
export LINGBOT_DIR=$SCRATCH/lingbot/dev/lingbot-vla-v2-route
RUNS=runs; [[ "$SMOKE" == --smoke ]] && RUNS=runs_smoke
OUT="$R2/$RUNS/$ARM"
[[ -f "$OUT/DONE" ]] && exit 0
YAML="$R2/configs/yaml/${ARM}${SMOKE:+_smoke}.yaml"
python3 $SCRATCH/lingbot/LingBotRoboTwin/route/make_yaml.py "$ARM" "$STEPS" "$SAVE" "$YAML" $SMOKE
if [[ "$ARM" == control ]]; then unset LINGBOT_ROUTE_CONFIG; else export LINGBOT_ROUTE_CONFIG="$R2/configs/$ARM.json"; fi
GLOBAL="${GLOBAL_BS:-128}"; MICRO="${MICRO_BS:-16}"; NGPU="${NGPU:-1}"; ACC=$(( GLOBAL / (MICRO * NGPU) ))
(( ACC >= 1 && ACC * MICRO * NGPU == GLOBAL )) || { echo "global $GLOBAL not divisible by micro $MICRO x $NGPU GPUs"; exit 2; }
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# same data sampling for every arm: each of the 4 datasets equally often (T1 = 2 datasets -> half the samples)
export LINGBOT_TASK_SAMPLING=uniform LINGBOT_SEEN_POOL=0 LINGBOT_FRAME_WEIGHTS=0
export LINGBOT_TIME_SAVE_MIN=15            # resume-only save every 15 min: preemptible B200s
export PYTHONPATH="$LINGBOT_DIR:$SCRATCH/lingbot/LingBotRoboTwin/src"
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])')
GC=false; [[ "${GRAD_CKPT:-0}" == 1 ]] && GC=true
FS=false; [[ "${FULL_SHARD:-0}" == 1 ]] && FS=true
cd "$LINGBOT_DIR"
bash $SCRATCH/lingbot/LingBotRoboTwin/scripts/hpc/run_in_env.sh \
  torchrun --nnodes=1 --nproc-per-node "$NGPU" --master-port "$PORT" tasks/vla/train_lingbotvla.py "$YAML" \
  --train.micro_batch_size "$MICRO" \
  --data.norm_stats_file $SCRATCH/lingbot/LingBotRoboTwin/configs/norm_stats/robotwin_clean.json \
  --train.gradient_accumulation_steps "$ACC" --train.global_batch_size "$GLOBAL" \
  --train.enable_gradient_checkpointing "$GC" --train.enable_full_shard "$FS" ${EXTRA_TRAIN_ARGS:-}
[[ -f "$OUT/checkpoints/global_step_$STEPS/hf_ckpt/model.safetensors.index.json" ]] || exit 3
cp "$R2/configs/$ARM.json" "$OUT/checkpoints/global_step_$STEPS/hf_ckpt/route_config.json" 2>/dev/null || true
touch "$OUT/checkpoints/global_step_$STEPS/keep" "$OUT/checkpoints/global_step_$STEPS/hf_ckpt/.ready" "$OUT/DONE"
