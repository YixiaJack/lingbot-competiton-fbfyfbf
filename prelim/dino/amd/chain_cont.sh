#!/usr/bin/env bash
# Start the next leg of an AMD continuation once the running one has finished, e.g. 3000 -> 5000 steps. Run from the
# package root:
#   NEXT_CFG=$PWD/configs/cont_amd_nodepth5k.yaml PREV_STEP=3000 bash dino/amd/chain_cont.sh   # detaches
# NEXT_CFG must have the same train.output_dir as the running leg: the trainer then resumes from its last checkpoint
# with optimizer, data-loader and RNG state (enable_resume). Waits for the process in <run>/train.pid to exit, then
# requires <run>/checkpoints/global_step_<PREV_STEP> to be complete (extra_state of every rank, model/ and optimizer/
# .metadata, every hf_ckpt shard, nothing written for 3 min; waits up to 30 min) and the clean-only norm stats; does
# not start otherwise (a crashed leg is left for a person to look at). Then waits for the GPUs to be free, starts
# dino/amd/train_amd.sh with NEXT_CFG, checks that the trainer loaded global_step_<PREV_STEP> and reports the first
# steps' LR, and makes sure dino/amd/mirror_ckpt.sh keeps mirroring the run. Log: <run>/chain.log, pid <run>/chain.pid.
set -uo pipefail
PKG="$(pwd)"
PY="${PY:-/opt/robotwin-env/bin/python}"
CFG="${NEXT_CFG:?NEXT_CFG=<yaml of the next leg>}"
PREV_STEP="${PREV_STEP:?PREV_STEP=<last step of the running leg>}"
NORM_SHA=92b37257234b37c8642894997847da496c6e6bb550c2968eced4439c8b6bed31  # configs/norm_stats/robotwin_clean.json
RUN=$("${PY}" -c "import yaml; print(yaml.safe_load(open('${CFG}'))['train']['output_dir'])")
if [[ "${CHAIN_CHILD:-0}" != 1 ]]; then
  CHAIN_CHILD=1 nohup setsid bash "$0" >> "${RUN}/chain.log" 2>&1 < /dev/null &
  echo $! > "${RUN}/chain.pid"; echo "chain started pid $(cat "${RUN}/chain.pid"), log ${RUN}/chain.log"
  exit 0
fi

log() { echo "$(date '+%F %T') $*"; }
complete() {  # checkpoint dir -> 0 if fully written (as dino/amd/mirror_ckpt.sh, plus every rank's extra_state)
  local c=$1 s r
  [[ -f "${c}/model/.metadata" && -f "${c}/optimizer/.metadata" && -f "${c}/hf_ckpt/model.safetensors.index.json" ]] || return 1
  for ((r = 0; r < ${NGPU:-8}; r++)); do [[ -s "${c}/extra_state/extra_state_rank_${r}.pt" ]] || return 1; done
  for s in $(python3 -c "import json,sys; print(' '.join(sorted(set(json.load(open(sys.argv[1]))['weight_map'].values()))))" \
             "${c}/hf_ckpt/model.safetensors.index.json"); do
    [[ -s "${c}/hf_ckpt/${s}" ]] || return 1
  done
  [[ -z "$(find "${c}" -newermt '3 minutes ago' -print -quit)" ]]
}
busy_gpus() { rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{gsub(/[^0-9]/,"",$NF); if ($NF > 2e9) n++} END{print n+0}'; }

PREV_PID=$(cat "${RUN}/train.pid")
PREV_LOG=$(ls -t "${PKG}"/runs/logs/cont_amd_*.log | head -1)
log "chain: wait for pid ${PREV_PID} (log ${PREV_LOG}), then step ${PREV_STEP} -> $(grep -E '^  max_steps:' "${CFG}" | awk '{print $2}') with ${CFG}"
while kill -0 "${PREV_PID}" 2>/dev/null; do sleep 60; done
log "pid ${PREV_PID} ended; last log line: $(tail -c 2000 "${PREV_LOG}" | tr '\r' '\n' | grep -v '^\s*$' | tail -1 | cut -c1-200)"

C="${RUN}/checkpoints/global_step_${PREV_STEP}"
for ((i = 0; i < 30; i++)); do complete "${C}" && break; sleep 60; done
complete "${C}" || { log "ERROR: ${C} is not complete after 30 min; not starting the next leg"; exit 1; }
newest=$(ls -d "${RUN}"/checkpoints/global_step_* | sed 's/.*global_step_//' | sort -n | tail -1)
[[ "${newest}" == "${PREV_STEP}" ]] || { log "ERROR: newest checkpoint is ${newest}, expected ${PREV_STEP}; not starting"; exit 1; }
sha=$(sha256sum "${PKG}/configs/norm_stats/robotwin_clean.json" | cut -d' ' -f1)
[[ "${sha}" == "${NORM_SHA}" ]] || { log "ERROR: norm stats sha256 ${sha} is not the clean-only file; not starting"; exit 1; }
log "step ${PREV_STEP} complete, norm stats clean-only"

for ((i = 0; i < 30; i++)); do (( $(busy_gpus) == 0 )) && break; sleep 60; done
(( $(busy_gpus) == 0 )) || { log "ERROR: $(busy_gpus) GPU(s) still hold memory after 30 min; not starting"; exit 1; }
CFG="${CFG}" bash dino/amd/train_amd.sh || { log "ERROR: train_amd.sh failed"; exit 1; }
NEW_LOG=$(ls -t "${PKG}"/runs/logs/cont_amd_*.log | head -1)
log "started pid $(cat "${RUN}/train.pid"), log ${NEW_LOG}"
envblock() { awk 'NR == 1 {next} /^    / {print; next} {exit}' "$1" | sort; }  # the env lines under the log's first line
d=$(diff <(envblock "${PREV_LOG}") <(envblock "${NEW_LOG}"))
[[ -z "${d}" ]] && log "launch environment identical to the previous leg" || log "launch environment differs from the previous leg: ${d}"

for ((i = 0; i < 40; i++)); do  # loading + the first step take ~5 min
  sleep 30
  kill -0 "$(cat "${RUN}/train.pid")" 2>/dev/null || { log "ERROR: training exited; see ${NEW_LOG}"; tail -c 3000 "${NEW_LOG}" | tr '\r' '\n' | grep -E 'Error|Traceback' | tail -5; exit 1; }
  step=$(tr '\r' '\n' < "${NEW_LOG}" | grep -oE "INFO - __main__ - Step [0-9]+/" | tail -1 | grep -oE '[0-9]+' | head -1)
  [[ -n "${step}" ]] && (( step >= PREV_STEP + 3 )) && break
done
loaded=$(grep -m1 -oE "Load distributed checkpoint from [^ ]+ successfully" "${NEW_LOG}")
log "${loaded:-WARNING: no 'Load distributed checkpoint' line in ${NEW_LOG}}"
[[ "${loaded}" == *"global_step_${PREV_STEP} "* ]] || log "WARNING: did not resume from global_step_${PREV_STEP}"
tr '\r' '\n' < "${NEW_LOG}" | grep "INFO - __main__ - Step" | tail -3 \
  | sed -E 's/.*(Step [0-9]+)\/[0-9]+.*(VLA_Loss [0-9.]+).*(LR [0-9.e-]+, Expert_LR [0-9.e-]+).*/    \1 \2 \3/'

mpid=$(cat "${RUN}/mirror.pid" 2>/dev/null)
if [[ -n "${mpid}" ]] && kill -0 "${mpid}" 2>/dev/null; then log "mirror pid ${mpid} still running (it follows train.pid)"
else bash dino/amd/mirror_ckpt.sh "${RUN}" | sed 's/^/    /'; fi
log "chain done"
