#!/usr/bin/env bash
# Once the AMD continuation is over (no dino/amd/chain_cont.sh and no training running for RUN), evaluate its newest
# complete hf_ckpt on the NVIDIA reference run's seeds (eval_amd/run_seedlist_eval.py) on GPUS (default all 8).
# With BASE=<hf_ckpt> (e.g. the 22k+8k starting point) GPUs 0-3 take the new checkpoint and GPUs 4-7 BASE instead, so
# both run on the same platform and scenes. First EPISODES_1 (7) episodes of every (setting, task), then on to
# EPISODES_2 (20). Run from the package root:
#   EXPECT=4000 bash eval_amd/after_train_eval.sh        # detaches; log <run>/after_eval.log, pid <run>/after_eval.pid
# Launcher logs: eval_runs/<run name>.launcher.log; results eval_runs/<run name>/ and /workspace/lingbot_eval/<run name>/.
set -uo pipefail
PKG="$(pwd)"
PY="${PY:-/opt/robotwin-env/bin/python}"
RUN="${RUN:-${PKG}/runs/cont8k_v3_nodepth3k_amd}"
BASE="${BASE:-}"  # 10/25 the 22k+8k copy on /workspace was deleted: no baseline by default
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
EXPECT="${EXPECT:-4000}"
TRAIN_CFG="${TRAIN_CFG:-${PKG}/configs/cont_amd_3kto4k_seed43.yaml}"
E1="${EPISODES_1:-7}"; E2="${EPISODES_2:-20}"
if [[ "${AFTER_CHILD:-0}" != 1 ]]; then
  AFTER_CHILD=1 nohup setsid bash "$0" >> "${RUN}/after_eval.log" 2>&1 < /dev/null &
  echo $! > "${RUN}/after_eval.pid"; echo "after_train_eval started pid $(cat "${RUN}/after_eval.pid"), log ${RUN}/after_eval.log"
  exit 0
fi

log() { echo "$(date '+%F %T') $*"; }
alive() {  # pid file, command pattern -> 0 if that process runs: not a zombie (PID 1 in this container does not reap
  local p        # them, so kill -0 succeeds after the exit; 10/25 the chain waited on one for hours), not a reused PID
  p=$(cat "$1" 2>/dev/null) || return 1
  [[ -n "${p}" && "$(ps -o stat= -p "${p}" 2>/dev/null)" != Z* ]] && tr '\0' ' ' < "/proc/${p}/cmdline" 2>/dev/null | grep -q -- "$2"
}
hf_complete() {  # hf_ckpt dir -> 0 if the index and every shard are there and nothing was written for 3 min
  local h=$1 s
  [[ -f "${h}/model.safetensors.index.json" ]] || return 1
  for s in $(python3 -c "import json,sys; print(' '.join(sorted(set(json.load(open(sys.argv[1]))['weight_map'].values()))))" \
             "${h}/model.safetensors.index.json" 2>/dev/null); do
    [[ -s "${h}/${s}" ]] || return 1
  done
  [[ -z "$(find "${h}" -newermt '3 minutes ago' -print -quit)" ]]
}
busy_gpus() { rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{gsub(/[^0-9]/,"",$NF); if ($NF > 2e9) n++} END{print n+0}'; }

log "wait until neither ${RUN}/chain.pid nor ${RUN}/train.pid runs"
while alive "${RUN}/chain.pid" chain_cont || alive "${RUN}/train.pid" train_lingbotvla; do sleep 60; done
log "training over; newest checkpoint dir: $(ls -d "${RUN}"/checkpoints/global_step_* | sed 's/.*global_step_//' | sort -n | tail -1)"
step=""
for ((i = 0; i < 30; i++)); do  # the last save may still be settling: wait for the newest one
  newest=$(ls -d "${RUN}"/checkpoints/global_step_* | sed 's/.*global_step_//' | sort -n | tail -1)
  hf_complete "${RUN}/checkpoints/global_step_${newest}/hf_ckpt" && { step=${newest}; break; }
  sleep 60
done
if [[ -z "${step}" ]]; then  # newest still incomplete after 30 min: take the newest complete one
  for st in $(ls -d "${RUN}"/checkpoints/global_step_* | sed 's/.*global_step_//' | sort -rn); do
    hf_complete "${RUN}/checkpoints/global_step_${st}/hf_ckpt" && { step=${st}; break; }
  done
fi
[[ -n "${step}" ]] || { log "ERROR: no complete hf_ckpt in ${RUN}"; exit 1; }
[[ "${step}" == "${EXPECT}" ]] || log "WARNING: evaluating step ${step}, not ${EXPECT}"
NEW="${RUN}/checkpoints/global_step_${step}/hf_ckpt"
for ((i = 0; i < 30; i++)); do (( $(busy_gpus) == 0 )) && break; sleep 60; done
(( $(busy_gpus) == 0 )) || { log "ERROR: $(busy_gpus) GPU(s) still hold memory after 30 min; not starting"; exit 1; }

mkdir -p "${PKG}/eval_runs"
side() {  # run name, model, gpus
  local e rc
  for e in "${E1}" "${E2}"; do
    log "$1: ${e} episodes, ${2}, GPUs $3"
    "${PY}" eval_amd/run_seedlist_eval.py --model-path "$2" --run-name "$1" --gpus "$3" --episodes "${e}" \
      --training-config "${TRAIN_CFG}" >> "${PKG}/eval_runs/$1.launcher.log" 2>&1
    rc=$?
    log "$1: ${e}-episode pass ended (rc ${rc}): $(grep -E 'demo_(clean|randomized): ' "${PKG}/eval_runs/$1.launcher.log" | tail -2 | tr '\n' ' ')"
  done
}
if [[ -n "${BASE}" ]]; then
  [[ -f "${BASE}/model.safetensors.index.json" ]] || { log "ERROR: BASE ${BASE} has no model.safetensors.index.json"; exit 1; }
  side "amd_s${step}" "${NEW}" 0,1,2,3 &
  sleep 30  # the launcher rewrites the patched client at start; do not let the two do it at the same moment
  side base_s22k8k_8k "${BASE}" 4,5,6,7 &
else
  side "amd_s${step}" "${NEW}" "${GPUS}" &
fi
wait
log "after_train_eval done"
