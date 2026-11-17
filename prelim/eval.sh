#!/usr/bin/env bash
# One-command evaluation of a checkpoint on all 50 tasks in one setting, on a single machine (no Slurm):
# one task per GPU at a time, 100 episodes each, seed 0, FP32 — the official protocol.
#   CKPT=/path/to/runs/main/checkpoints/global_step_XXXX/hf_ckpt TASK_CONFIG=demo_clean \
#   LINGBOT_DIR=... ROBOTWIN_DIR=... CKPT_DIR=... OUT=/path/to/eval/main bash eval.sh
#   (then TASK_CONFIG=demo_randomized; then python3 src/make_results_json.py --eval_dir $OUT ...)
# Needs conda envs `lingbotvla` and `RoboTwin` (see README); CONDA_SH / SIM_CONDA_SH point at their conda.sh.
# CKPT_DIR holds the downloaded base models (Qwen3-VL-4B-Instruct, lingbot-vla-v2-6b, ...). The checkpoint's training
# config has our cluster paths; src/stage_ckpt.py writes a localized copy under $OUT/_staged (norm stats pinned to the
# clean-only file and SHA-256 checked) and the server is pointed at that. LINGBOT_DIR must be set up by
# scripts/setup_lingbot.sh (the DINO-fusion model needs the patched code).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CKPT:?}" "${TASK_CONFIG:?}" "${LINGBOT_DIR:?}" "${ROBOTWIN_DIR:?}" "${CKPT_DIR:?}" "${OUT:?}"
CKPT=$(python3 "${HERE}/src/stage_ckpt.py" --ckpt "${CKPT}" --ckpt_dir "${CKPT_DIR}" --stage "${OUT}/_staged") || exit 1
if grep -qE "^\s*dino_fusion_path:\s*/" "$(dirname "$(dirname "$(dirname "${CKPT}")")")/lingbotvla_cli.yaml" \
   && ! grep -q dino_fusion "${LINGBOT_DIR}/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"; then
  echo "DINO-fusion checkpoint but ${LINGBOT_DIR} is not patched: VARIANT=dino bash scripts/setup_lingbot.sh" >&2; exit 1
fi
echo "serving ${CKPT}"
export WORK_DIR="${HERE}" LINGBOT_DIR ROBOTWIN_DIR PROJ="${PROJ:-$(dirname "${OUT}")}"
NGPU=$(nvidia-smi -L | wc -l)
mapfile -t TASKS < <(python3 -c "import json;print('\n'.join(json.load(open('${HERE}/configs/tasks50.json'))))")
run_gpu() {
  local g=$1
  for ((i=g; i<${#TASKS[@]}; i+=NGPU)); do
    t=${TASKS[$i]}
    [[ -f "${OUT}/${TASK_CONFIG}/${t}.result" ]] && continue
    CUDA_VISIBLE_DEVICES=$g bash "${HERE}/src/eval_one_task.sh" "${CKPT}" "${t}" "${TASK_CONFIG}" "${OUT}/${TASK_CONFIG}" $((9400 + g))
  done
}
for ((g=0; g<NGPU; g++)); do run_gpu $g & done; wait
ls "${OUT}/${TASK_CONFIG}"/*.result | wc -l
