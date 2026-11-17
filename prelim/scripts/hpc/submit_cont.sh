#!/usr/bin/env bash
# Second stage of the from-scratch strong-augmentation run: continue from strong_scratch24k's BASE_STEP checkpoint
# for STEPS more steps (default 22000 + 8000 = 30k total; the first stage was preempted at 22.5k and its B200
# requeue was hours away, so the planned 24k + 6k became 22k + 8k).
#   lr 2e-5 -> 5e-6 (1/5 .. 1/20 of the first stage), 3% warmup, checkpoint every 1000 steps (pick the best later)
#   stronger whitelist augmentation + Fourier amplitude modulation + wrist-camera dropout p=0.4 (configs/aug_strong.json)
#   task-uniform sampling with low-motion frames down-weighted (meta/frame_weights.npy), seen-pool instructions,
#   proprio noise 0.01, ViT lr x0.1; the upstream patch is applied at job start (APPLY_PATCH=1)
# One run directory, three jobs with the same --job-name: a 4xA100 job starts the run now, and a 4xH100 and a 4xB200
# job with HANDOFF=1 take it over as soon as either gets GPUs (cancel the others, resume from the newest checkpoint;
# see train.sbatch). All use micro 16 x 2-step accumulation = global 128 so dataloader / optimizer state carry over.
# Also submits the wandb mirror and the HF upload of the base checkpoint. DEP=<job id>: wait for that job to succeed.
#   BASE_STEP=22000 STEPS=8000 scripts/hpc/submit_cont.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
W="${PWD}"
source scripts/hpc/env.sh
BASE_STEP="${BASE_STEP:-22000}"
STEPS="${STEPS:-8000}"
BASE="${RUN_DIR}/strong_scratch24k/checkpoints/global_step_${BASE_STEP}/hf_ckpt"
[[ -n "${DEP:-}" || -f "${BASE}/model.safetensors.index.json" ]] || { echo "missing ${BASE}" >&2; exit 1; }
DEPARG=()
[[ -n "${DEP:-}" ]] && DEPARG=(--dependency=afterok:${DEP})
RUN="strong_scratch$((BASE_STEP / 1000))k_cont$((STEPS / 1000))k"

export APPLY_PATCH=1 PYTHONPATH="${W}/src" LINGBOT_AUG_CONFIG="${W}/configs/aug_strong.json" LINGBOT_SEEN_POOL=1 \
       LINGBOT_STATE_NOISE=0.01 LINGBOT_VISION_LR_SCALE=0.1 LINGBOT_TASK_SAMPLING=uniform LINGBOT_FRAME_WEIGHTS=1
ARGS="--model.model_path ${BASE} --data.image_augment true --train.lr 2e-5 --train.lr_min 5e-6 --train.lr_warmup_ratio 0.03"
submit() {  # gpu_label time handoff gpu_opts...
  local label=$1 time=$2 handoff=$3; shift 3
  HANDOFF=${handoff} RUN_NAME=${RUN} MAX_STEPS=${STEPS} SAVE_STEPS=1000 MICRO_BS=16 GLOBAL_BS=128 TRAIN_ARGS="${ARGS}" \
    scripts/hpc/submit.sh scripts/hpc/train.sbatch --parsable --job-name="lb_${RUN}" --time="${time}" \
    ${DEPARG[@]+"${DEPARG[@]}"} "$@" | tail -1
}
A=$(submit a100 24:00:00 0 --gres=gpu:a100:4 --constraint=a100 --cpus-per-task=48 --mem=400G)
H=$(submit h100 16:00:00 1 --gres=gpu:4 --constraint=h100 --cpus-per-task=48 --mem=400G)
B=$(submit b200 08:00:00 1 --gres=gpu:b200:4 --constraint=b200 --cpus-per-task=48 --mem=400G)
unset APPLY_PATCH PYTHONPATH LINGBOT_AUG_CONFIG LINGBOT_SEEN_POOL LINGBOT_STATE_NOISE LINGBOT_VISION_LR_SCALE \
      LINGBOT_TASK_SAMPLING LINGBOT_FRAME_WEIGHTS
# one mirror for the shared run directory, alive while any job of the group is queued (TRAIN_JOB_NAME)
RUN_NAME=${RUN} TRAIN_JOB_NAME="lb_${RUN}" scripts/hpc/submit.sh --cpu scripts/hpc/wandb_sync.sbatch >/dev/null
U=skipped
if [[ "${UPLOAD:-1}" == 1 ]]; then
  U=$(RUN_NAME=strong_scratch24k STEP=${BASE_STEP} REPO=${HF_REPO_PREFIX:?set HF_REPO_PREFIX (private HF namespace) or UPLOAD=0}/lingbot-robotwin-strong-scratch$((BASE_STEP / 1000))k \
      scripts/hpc/submit.sh --cpu scripts/hpc/stage_and_upload.sbatch --parsable ${DEPARG[@]+"${DEPARG[@]}"} | tail -1)
fi
echo "${RUN}: a100=${A} h100=${H} b200=${B} (job name lb_${RUN}) | upload base ${BASE_STEP}: ${U}"
