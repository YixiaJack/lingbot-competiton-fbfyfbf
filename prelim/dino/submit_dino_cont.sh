#!/usr/bin/env bash
# DINOv2-fusion continuation: base = strong_scratch22k_cont8k / global_step_8000 + frozen DINOv2 (zero-init projection,
# built by dino/make_dino_init.py into dev/runs/dino_init_cont8k_8000). 6k steps, checkpoint every 500 (SAVE): at ~1.9 s/step
# 1000 steps take ~31 min, about the 30-min preemption window, so a preemption could lose everything every time.
#   main lr 1e-5 -> 5e-6 (3% warmup), dino_fusion layers x10 (LINGBOT_DINO_LR_SCALE), ViT x0.1, same aug / sampling as the
#   +8k continuation. Code: the DINO dev checkout (dev/lingbot-vla-v2-dino, patched by src/patch_upstream.py and
#   dino/patch_dino.py); the live lingbot-vla-v2 checkout is not used or modified.
# One 4xB200 and one 4xH100 job with the same --job-name, ranked handoff (see below).
# A100=1 / L40S=1 add a 2xA100 / 4xL40S member (rank 1: keeps the run going until a B200/H100 member takes over).
# MAX_STEPS (default 6000) can be raised for a running group: resubmit with the same RUN, the jobs resume from the newest
# checkpoint (note: the cosine schedule is defined over MAX_STEPS, so raising it lifts the lr from that step on).
# LR / LR_MIN changed on a resubmission also apply from the resumed step (dino/patch_dino.py resume_lr; without it the
# checkpoint's lr was kept: the 10/24 resubmission with LR=5e-6 trained at 1e-5 from step 500).
# SIDECAR=0 skips the wandb sidecar (one is already following the job name).
# MEMBERS (default "b200 h100", plus a100 / l40s with A100=1 / L40S=1) picks the members to submit, e.g. MEMBERS=a100
# SIDECAR=0 to replace one failed member of a running group with the same settings.
# Micro batch per GPU: 16 on B200 / H100; 8 on 2xA100 (MICRO_A100) and 4 on 4xL40S (MICRO_L40S), both with gradient
# checkpointing + full sharding. Only the global batch (128) has to match across the group: the dataloader yields
# global / n_gpu samples per rank and step whatever the micro batch is, so resuming is unaffected.
# AUG (default configs/aug_strong.json): augmentation config, e.g. configs/aug_strong_v2.json (p_identity 0.15).
# LR / LR_MIN (default 1e-5 / 5e-6): main learning rate schedule (DINO layers x LINGBOT_DINO_LR_SCALE, ViT x0.1).
# TASK_SAMPLING (default uniform): e.g. temp:2 for temperature sampling over tasks (src/lingbot_sampling.py).
#   dino/submit_dino_cont.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
W="${PWD}"
source scripts/hpc/env.sh
DEV="${PROJ}/dev/lingbot-vla-v2-dino"
INIT="${PROJ}/dev/runs/dino_init_cont8k_8000/checkpoints/global_step_0/hf_ckpt"
DINO_DIR="${CKPT_DIR}/dinov2-with-registers-base"
RUN="${RUN:-dino_cont8k_6k}"
[[ -f "${INIT}/model.safetensors.index.json" ]] || { echo "missing ${INIT} (dino/make_dino_init.py)" >&2; exit 1; }
grep -q "dino_fusion" "${DEV}/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py" || { echo "dev checkout not patched" >&2; exit 1; }

export LINGBOT_DIR="${DEV}" APPLY_PATCH=0 HANDOFF=0 PYTHONPATH="${DEV}:${W}/src" \
       LINGBOT_AUG_CONFIG="${W}/${AUG:-configs/aug_strong.json}" LINGBOT_SEEN_POOL=1 LINGBOT_STATE_NOISE=0.01 \
       LINGBOT_VISION_LR_SCALE=0.1 LINGBOT_TASK_SAMPLING="${TASK_SAMPLING:-uniform}" LINGBOT_FRAME_WEIGHTS=1 LINGBOT_DINO_LR_SCALE=10
ARGS="--model.model_path ${INIT} --train.dino_fusion_path ${DINO_DIR} --data.image_augment true --train.lr ${LR:-1e-5} --train.lr_min ${LR_MIN:-5e-6} --train.lr_warmup_ratio 0.03"
# gradient checkpointing + full sharding (same checkpoint layout under FSDP2) for the small-memory members
LOWMEM="--train.enable_gradient_checkpointing true --train.enable_full_shard true"
submit() {  # time gpu_opts...
  local time=$1; shift
  RUN_NAME=${RUN} MAX_STEPS=${MAX_STEPS:-6000} SAVE_STEPS=${SAVE:-500} MICRO_BS=${MICRO:-16} GLOBAL_BS=128 TRAIN_ARGS="${ARGS} ${EXTRA_TRAIN_ARGS:-}" \
    scripts/hpc/submit.sh scripts/hpc/train.sbatch --parsable --job-name="lb_${RUN}" --time="${time}" "$@" | tail -1
}
# ranked handoff (scripts/hpc/train.sbatch HANDOFF_RANK): B200 3 > H100 2 > A100 = L40S 1; a faster member that starts
# requeues a slower running one (it stays queued as a backup); equal ranks never run at the same time
if [[ -z "${MEMBERS:-}" ]]; then
  MEMBERS="b200 h100"
  if [[ "${A100:-0}" == 1 ]]; then MEMBERS+=" a100"; fi
  if [[ "${L40S:-0}" == 1 ]]; then MEMBERS+=" l40s"; fi
fi
has() { [[ " ${MEMBERS} " == *" $1 "* ]]; }
B=none; H=none; A=none; L=none
if has b200; then B=$(HANDOFF_RANK=3 submit 08:00:00 --gres=gpu:b200:4 --constraint=b200 --cpus-per-task=48 --mem=400G); fi
if has h100; then H=$(HANDOFF_RANK=2 submit 12:00:00 --gres=gpu:4 --constraint=h100 --cpus-per-task=48 --mem=400G); fi
if has a100; then  # 2x80 GB: micro 16 without LOWMEM ran out of memory in the first step (10/23)
  A=$(HANDOFF_RANK=1 SAVE=${SAVE_A100:-200} MICRO=${MICRO_A100:-8} EXTRA_TRAIN_ARGS="${LOWMEM}" \
      submit 12:00:00 --gres=gpu:2 --constraint=a100 --cpus-per-task=24 --mem=200G)
fi
if has l40s; then  # 4x48 GB
  L=$(HANDOFF_RANK=1 SAVE=${SAVE_L40S:-200} MICRO=${MICRO_L40S:-4} EXTRA_TRAIN_ARGS="${LOWMEM}" \
      submit 12:00:00 --gres=gpu:l40s:4 --constraint=l40s --cpus-per-task=48 --mem=400G)
fi
unset HANDOFF APPLY_PATCH PYTHONPATH LINGBOT_AUG_CONFIG LINGBOT_SEEN_POOL LINGBOT_STATE_NOISE LINGBOT_VISION_LR_SCALE \
      LINGBOT_TASK_SAMPLING LINGBOT_FRAME_WEIGHTS LINGBOT_DINO_LR_SCALE
[[ "${SIDECAR:-1}" == 1 ]] && RUN_NAME=${RUN} TRAIN_JOB_NAME="lb_${RUN}" scripts/hpc/submit.sh --cpu scripts/hpc/wandb_sync.sbatch >/dev/null
echo "${RUN}: lr=${LR:-1e-5}->${LR_MIN:-5e-6} aug=${AUG:-configs/aug_strong.json} sampling=${TASK_SAMPLING:-uniform} max_steps=${MAX_STEPS:-6000} b200=${B} h100=${H} a100=${A} l40s=${L} (job name lb_${RUN}, run dir ${RUN_DIR}/${RUN})"
