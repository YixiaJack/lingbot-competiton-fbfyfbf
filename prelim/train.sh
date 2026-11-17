#!/usr/bin/env bash
# One-command training of the submitted model strong_scratch22k_cont8k / global_step_8000 on a single multi-GPU machine
# (no Slurm), in its two stages (scripts/hpc/submit_strong.sh and scripts/hpc/submit_cont.sh are the cluster versions):
#   stage 1  strong_scratch24k: from robbyant/lingbot-vla-v2-6b, whitelist augmentation (configs/aug_whitelist.json),
#            seen-pool instructions, proprio noise 0.01, ViT lr x0.1; lr 1e-4 -> 5e-5 cosine over 24,000 steps, no warmup,
#            micro 32 x 4 GPUs = global 128. Stage 2 starts from its global_step_22000 (the original run was preempted at
#            22,487 and not resumed; running it to 24,000 changes nothing for stage 2).
#   stage 2  strong_scratch22k_cont8k: from stage 1's global_step_22000, 8,000 steps, lr 2e-5 -> 5e-6 cosine, 3% warmup,
#            stronger augmentation (configs/aug_strong.json), task-uniform sampling with low-motion frames down-weighted,
#            seen-pool instructions, proprio noise 0.01, ViT lr x0.1; micro 16 x 4 GPUs x 2 = global 128.
#   LINGBOT_DIR=/path/to/lingbot-vla-v2 CKPT_DIR=/path/to/ckpts DATA_DIR=/path/to/data/lerobot/robotwin_clean \
#   RAW_DIR=/path/to/data/raw OUTPUT_DIR=/path/to/runs bash train.sh          # STAGE=1 or STAGE=2 runs one stage
# Needs: LINGBOT_DIR prepared by scripts/setup_lingbot.sh (upstream @ be969b8 + src/patch_upstream.py, whose changes act
# only when the LINGBOT_* variables below are set); the converted clean datasets; meta/seen_instructions.json
# (src/make_seen_pool.py, needs RAW_DIR) and meta/frame_weights.npy (src/make_frame_weights.py) in every dataset, which
# this script writes if missing. Paths inside configs/*.yaml / configs/robotwin_clean.txt point at our cluster
# ($SCRATCH/lingbot/...); they are rewritten to the variables above in a copy under $OUTPUT_DIR/_cfg.
# The global batch stays 128 with other GPU counts (gradient accumulation); on 80 GB cards use MICRO_BS=16 for stage 1.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LINGBOT_DIR:?}" "${CKPT_DIR:?}" "${DATA_DIR:?}" "${OUTPUT_DIR:?}"
STAGE="${STAGE:-all}"
GLOBAL_BS=128
NGPU=$(nvidia-smi -L | wc -l)
OUT1="${OUTPUT_DIR}/strong_scratch24k"; OUT2="${OUTPUT_DIR}/strong_scratch22k_cont8k"
INIT2="${OUT1}/checkpoints/global_step_22000/hf_ckpt"

grep -q "LINGBOT_AUG_CONFIG" -r "${LINGBOT_DIR}/lingbotvla/data/vla_data/" \
  || { echo "${LINGBOT_DIR} is not patched: DEST=${LINGBOT_DIR} bash scripts/setup_lingbot.sh" >&2; exit 1; }
for ds in "${DATA_DIR}"/*-aloha-agilex_clean_50-50; do
  if [[ ! -f "${ds}/meta/seen_instructions.json" ]]; then
    : "${RAW_DIR:?meta/seen_instructions.json missing: set RAW_DIR (raw RoboTwin clean data) to build it}"
    python3 "${HERE}/src/make_seen_pool.py" --raw_root "${RAW_DIR}" --lerobot_root "${DATA_DIR}"; break
  fi
done
for ds in "${DATA_DIR}"/*-aloha-agilex_clean_50-50; do
  [[ -f "${ds}/meta/frame_weights.npy" ]] || { python3 "${HERE}/src/make_frame_weights.py" --lerobot_root "${DATA_DIR}"; break; }
done

CFG="${OUTPUT_DIR}/_cfg"; mkdir -p "${CFG}/robot_configs" "${CFG}/norm_stats"
OLD='\$SCRATCH/lingbot'
sub() { sed -e "s#${OLD}/ckpts#${CKPT_DIR}#g" -e "s#${OLD}/data/lerobot/robotwin_clean#${DATA_DIR}#g" \
            -e "s#${OLD}/LingBotRoboTwin/configs#${CFG}#g" -e "s#${OLD}/runs/robotwin_clean#${OUTPUT_DIR}#g" "$1"; }
sub "${HERE}/configs/robotwin_clean.yaml" > "${CFG}/robotwin_clean.yaml"
sub "${HERE}/configs/robotwin_clean.txt" > "${CFG}/robotwin_clean.txt"
sub "${HERE}/configs/robot_configs/robotwin_clean.yaml" > "${CFG}/robot_configs/robotwin_clean.yaml"
cp "${HERE}/configs/norm_stats/robotwin_clean.json" "${CFG}/norm_stats/"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${HERE}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LINGBOT_SEEN_POOL=1 LINGBOT_STATE_NOISE=0.01 LINGBOT_VISION_LR_SCALE=0.1

train() {  # output_dir micro max_steps save_steps extra args...
  local out=$1 micro=$2 steps=$3 save=$4; shift 4
  local acc=$(( GLOBAL_BS / (micro * NGPU) ))
  (( acc >= 1 && acc * micro * NGPU == GLOBAL_BS )) || { echo "global ${GLOBAL_BS} not divisible by ${micro} x ${NGPU}" >&2; exit 1; }
  (cd "${LINGBOT_DIR}" && torchrun --nnodes=1 --nproc-per-node "${NGPU}" --master-port="${MASTER_PORT:-29531}" \
    tasks/vla/train_lingbotvla.py "${CFG}/robotwin_clean.yaml" \
    --data.norm_stats_file "${CFG}/norm_stats/robotwin_clean.json" \
    --train.output_dir "${out}" \
    --train.micro_batch_size "${micro}" --train.global_batch_size "${GLOBAL_BS}" --train.gradient_accumulation_steps "${acc}" \
    --train.max_steps "${steps}" --train.save_steps "${save}" --data.image_augment true "$@")
}

if [[ "${STAGE}" == all || "${STAGE}" == 1 ]]; then
  LINGBOT_AUG_CONFIG="${HERE}/configs/aug_whitelist.json" \
    train "${OUT1}" "${MICRO_BS:-32}" 24000 1000
fi
if [[ "${STAGE}" == all || "${STAGE}" == 2 ]]; then
  [[ -f "${INIT2}/model.safetensors.index.json" ]] || { echo "missing ${INIT2} (stage 1)" >&2; exit 1; }
  LINGBOT_AUG_CONFIG="${HERE}/configs/aug_strong.json" LINGBOT_TASK_SAMPLING=uniform LINGBOT_FRAME_WEIGHTS=1 \
    train "${OUT2}" 16 8000 1000 \
    --model.model_path "${INIT2}" --train.lr 2e-5 --train.lr_min 5e-6 --train.lr_warmup_ratio 0.03
fi
