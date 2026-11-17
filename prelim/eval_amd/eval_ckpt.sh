#!/usr/bin/env bash
# Evaluate one hf_ckpt on the AMD box right away (no waiting for a training run, unlike eval_amd/after_train_eval.sh):
# eval_amd/run_seedlist_eval.py on GPUS with the NVIDIA 22k+8k run's seeds (its default seed list), EPISODES per
# (setting, task), randomized scenes first. Logs in the format of after_train_eval.sh, so scripts/amd_followup.sh on the
# Mac notifies and copies eval_runs/ to the Mac. Run from the package root:
#   NAME=amd_s3000 CKPT=runs/cont8k_v3_nodepth3k_amd/checkpoints/global_step_3000/hf_ckpt EPISODES=7 bash eval_amd/eval_ckpt.sh
# Detaches; log <run>/after_eval.log, pid <run>/eval_ckpt.pid; launcher log eval_runs/<NAME>.launcher.log.
# TRAIN_CFG only supplies data.norm_stats_file (clean-only stats); its model_path must exist on this box.
set -uo pipefail
PKG="$(pwd)"
PY="${PY:-/opt/robotwin-env/bin/python}"
RUN="${RUN:-${PKG}/runs/cont8k_v3_nodepth3k_amd}"
NAME="${NAME:?NAME=<eval run name>}"
CKPT="$(realpath "${CKPT:?CKPT=<hf_ckpt dir>}")"
EPISODES="${EPISODES:-7}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TRAIN_CFG="${TRAIN_CFG:-${PKG}/configs/cont_amd_3kto4k_seed43.yaml}"
if [[ "${EVAL_CHILD:-0}" != 1 ]]; then
  EVAL_CHILD=1 NAME="${NAME}" CKPT="${CKPT}" EPISODES="${EPISODES}" GPUS="${GPUS}" TRAIN_CFG="${TRAIN_CFG}" \
    nohup setsid bash "$0" >> "${RUN}/after_eval.log" 2>&1 < /dev/null &
  echo $! > "${RUN}/eval_ckpt.pid"; echo "eval_ckpt started pid $(cat "${RUN}/eval_ckpt.pid"), log ${RUN}/after_eval.log"
  exit 0
fi

log() { echo "$(date '+%F %T') $*"; }
busy_gpus() { rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{gsub(/[^0-9]/,"",$NF); if ($NF > 2e9) n++} END{print n+0}'; }
[[ -f "${CKPT}/model.safetensors.index.json" ]] || { log "ERROR: ${CKPT} has no model.safetensors.index.json"; exit 1; }
[[ -f "$("${PY}" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['model']['model_path'])" "${TRAIN_CFG}")/config.json" ]] \
  || { log "ERROR: model_path of ${TRAIN_CFG} does not exist here"; exit 1; }
(( $(busy_gpus) == 0 )) || { log "ERROR: $(busy_gpus) GPU(s) hold memory; not starting"; exit 1; }
mkdir -p "${PKG}/eval_runs"
log "${NAME}: ${EPISODES} episodes, ${CKPT}, GPUs ${GPUS}"
"${PY}" eval_amd/run_seedlist_eval.py --model-path "${CKPT}" --run-name "${NAME}" --gpus "${GPUS}" --episodes "${EPISODES}" \
  --configs demo_randomized,demo_clean --training-config "${TRAIN_CFG}" >> "${PKG}/eval_runs/${NAME}.launcher.log" 2>&1
rc=$?
log "${NAME}: ${EPISODES}-episode pass ended (rc ${rc}): $(grep -E 'demo_(clean|randomized): ' "${PKG}/eval_runs/${NAME}.launcher.log" | tail -2 | tr '\n' ' ')"
log "after_train_eval done"
