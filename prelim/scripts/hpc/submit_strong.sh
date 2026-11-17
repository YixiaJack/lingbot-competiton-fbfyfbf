#!/usr/bin/env bash
# Submit the two whitelist-augmentation runs, each as a B200 job plus an H100/H200 "twin" with identical training
# settings (train.sbatch picks micro 32 on >=120 GB cards, 16 x 2-step accumulation on 80 GB cards; global 128). Whichever of a pair
# starts first is kept; cancel the other. Also starts a wandb mirror per job.
#   continue: from the main run's 40k checkpoint, 10k steps, lr 5e-5 -> 2e-5, save every 1000 (preemption loss)
#   scratch:  from lingbot-vla-v2-6b, 24k steps, lr 1e-4 -> 5e-5 (config default), save every 1000;
#             it waits (--dependency after) until the continue run has started, so continue gets the first slot
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
W="${PWD}"
source scripts/hpc/env.sh
export PYTHONPATH="${W}/src" LINGBOT_AUG_CONFIG="${W}/configs/aug_whitelist.json" LINGBOT_SEEN_POOL=1 \
       LINGBOT_STATE_NOISE=0.01 LINGBOT_VISION_LR_SCALE=0.1
MAIN40K="${RUN_DIR}/main_b200x4/checkpoints/global_step_40000/hf_ckpt"
CONT_ARGS="--model.model_path ${MAIN40K} --data.image_augment true --train.lr 5e-5 --train.lr_min 2e-5 --train.lr_warmup_ratio 0.05"
SCRATCH_ARGS="--data.image_augment true"
B200=(--gres=gpu:b200:4 --constraint=b200 --cpus-per-task=48 --mem=400G)
HOPPER=(--gres=gpu:4 "--constraint=h100|h200" --cpus-per-task=48 --mem=400G)

submit() {  # name run_name max_steps save_steps micro args time gpu_opts...
  local name=$1 run=$2 steps=$3 save=$4 micro=$5 args=$6 time=$7; shift 7
  RUN_NAME=${run} MAX_STEPS=${steps} SAVE_STEPS=${save} MICRO_BS=${micro} GLOBAL_BS=128 TRAIN_ARGS="${args}" \
    scripts/hpc/submit.sh scripts/hpc/train.sbatch --parsable --job-name="${name}" --time="${time}" "$@" | tail -1
}
C=$(submit lb_strong_c strong_main40k_10k 10000 1000 32 "${CONT_ARGS}" 12:00:00 "${B200[@]}")
CH=$(submit lb_strong_c_h strong_main40k_10k_h 10000 1000 auto "${CONT_ARGS}" 24:00:00 "${HOPPER[@]}")
S=$(submit lb_strong_s strong_scratch24k 24000 1000 32 "${SCRATCH_ARGS}" 24:00:00 "${B200[@]}" "--dependency=after:${C}?after:${CH}")
SH=$(submit lb_strong_s_h strong_scratch24k_h 24000 1000 auto "${SCRATCH_ARGS}" 36:00:00 "${HOPPER[@]}" "--dependency=after:${C}?after:${CH}")
unset PYTHONPATH
for pair in "strong_main40k_10k:${C}" "strong_main40k_10k_h:${CH}" "strong_scratch24k:${S}" "strong_scratch24k_h:${SH}"; do
  RUN_NAME=${pair%%:*} TRAIN_JOB=${pair##*:} scripts/hpc/submit.sh --cpu scripts/hpc/wandb_sync.sbatch >/dev/null
done
echo "continue: b200=${C} hopper=${CH} | scratch: b200=${S} hopper=${SH}"
