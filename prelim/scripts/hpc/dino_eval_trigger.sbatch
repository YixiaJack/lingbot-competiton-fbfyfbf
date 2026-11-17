#!/usr/bin/env bash
# CPU sidecar: freeze a DINO checkpoint for evaluation once the DINO training has finished or stalled.
#   FREEZE_AT="2026-10-25 10:00 +0800" OLD_ARRAYS="<array id> <array id>" scripts/hpc/submit.sh --cpu scripts/hpc/dino_eval_trigger.sbatch
#   (current run: also EVAL_DONE=q100_s22k8k_8k:100)
# Every 5 min: if the newest checkpoint of RUN_DIR/<DINO_RUN> reached DINO_MAX_STEPS, or no job named DINO_JOB_NAME has
# been running for STALL_MIN minutes (preempted and not rescheduled), or the time is past FREEZE_AT (any `date -d` time;
# the training keeps running), or every item of EVAL_DONE="<eval name>:<episodes>" (both configs x 50 tasks) has a
# .result with that many episodes (the eval pool would otherwise sit idle), or RUN_DIR/dino_eval/FREEZE_NOW exists
# (touch it to freeze by hand), copy the newest complete hf_ckpt (index + every shard, nothing written for 3 min, sizes
# checked) to RUN_DIR/dino_eval/run/checkpoints/global_step_<N>/hf_ckpt with the run's lingbotvla_cli.yaml
# at RUN_DIR/dino_eval/run/ (the policy server reads <ckpt>/../../../lingbotvla_cli.yaml), point
# RUN_DIR/dino_eval/run/checkpoints/current at it and write hf_ckpt/.ready. The eval pool's DINO phase uses
# .../checkpoints/current/hf_ckpt and starts once .ready exists. The copy is independent of the training run, whose
# pruning keeps deleting old checkpoints. Freezes once; later checkpoints are not evaluated.
# OLD_ARRAYS: eval arrays (or an earlier trigger) replaced by a newer one; their requeued (PENDING) elements are cancelled
# every round.
# RUN_DIR/dino_eval/trigger.env (shell assignments of STALL_MIN, MAX_STEPS, FREEZE_AT, EVAL_DONE) is re-read every round,
# so a running trigger can be retuned without resubmitting it.
#SBATCH --job-name=lb_dinotrig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/%u/lingbot/logs/dinotrig_%j.log
set -uo pipefail
cd "${WORK_DIR:-/scratch/${USER}/lingbot/LingBotRoboTwin}"
source scripts/hpc/env.sh
RUN="${RUN_DIR}/${DINO_RUN:-dino_cont8k_6k}"
JOB_NAME="${DINO_JOB_NAME:-lb_dino_cont8k_6k}"
MAX_STEPS="${DINO_MAX_STEPS:-15000}"
STALL_MIN="${STALL_MIN:-60}"
FREEZE_AT_S=0; [[ -n "${FREEZE_AT:-}" ]] && FREEZE_AT_S=$(date -d "${FREEZE_AT}" +%s)
EVAL_DONE="${EVAL_DONE:-}"
OVERRIDES=""
mapfile -t ALL_TASKS < <(python3 -c "import json; print('\n'.join(json.load(open('configs/tasks50.json'))))")
E="${RUN_DIR}/dino_eval"; OUT="${E}/run"; STATE="${E}/last_running"
mkdir -p "${E}"
log() { echo "$(date '+%F %T') $*"; }

hf_complete() {  # hf_ckpt dir -> 0 if the index and every shard it names are there and nothing was written for 3 min
  local h=$1 s
  [[ -f "${h}/model.safetensors.index.json" ]] || return 1
  for s in $(python3 -c "import json,sys; print(' '.join(sorted(set(json.load(open(sys.argv[1]))['weight_map'].values()))))" \
             "${h}/model.safetensors.index.json" 2>/dev/null); do
    [[ -s "${h}/${s}" ]] || return 1
  done
  [[ -z "$(find "${h}" -newermt '3 minutes ago' -print -quit)" ]]
}
newest_step() { ls -d "${RUN}"/checkpoints/global_step_* 2>/dev/null | sed 's/.*global_step_//' | grep -E '^[0-9]+$' | sort -n | tail -1; }
newest_hf_step() {
  local st
  for st in $(ls -d "${RUN}"/checkpoints/global_step_* 2>/dev/null | sed 's/.*global_step_//' | grep -E '^[0-9]+$' | sort -rn); do
    hf_complete "${RUN}/checkpoints/global_step_${st}/hf_ckpt" && { echo "${st}"; return 0; }
  done
  return 1
}

eval_done() {  # 0 once every (config, task) item of EVAL_DONE has a .result reporting at least that many episodes
  local name=${EVAL_DONE%%:*} n=${EVAL_DONE##*:} cfg t r
  for cfg in demo_clean demo_randomized; do
    for t in "${ALL_TASKS[@]}"; do
      r=$(awk '{print $2}' "${PROJ}/eval/${name}/${cfg}/${t}.result" 2>/dev/null)
      [[ -n "${r}" ]] && (( r >= n )) || return 1
    done
  done
}

load_overrides() {  # re-read RUN_DIR/dino_eval/trigger.env; log it when it changes
  [[ -f "${E}/trigger.env" ]] || return 0
  local now; now=$(cat "${E}/trigger.env")
  source "${E}/trigger.env"
  FREEZE_AT_S=0; [[ -n "${FREEZE_AT:-}" ]] && FREEZE_AT_S=$(date -d "${FREEZE_AT}" +%s)
  [[ "${now}" == "${OVERRIDES}" ]] && return 0
  OVERRIDES="${now}"
  log "trigger.env: stall ${STALL_MIN} min, max steps ${MAX_STEPS}, freeze at ${FREEZE_AT:-never}, eval ${EVAL_DONE:-(none)}"
}

freeze() {  # reason
  local st src dst
  st=$(newest_hf_step) || { log "no complete hf_ckpt in ${RUN} yet"; return 1; }
  src="${RUN}/checkpoints/global_step_${st}/hf_ckpt"; dst="${OUT}/checkpoints/global_step_${st}/hf_ckpt"
  local tmp="${dst}.partial_${SLURM_JOB_ID:-$$}"  # own name: an earlier trigger still running may copy at the same time
  log "freeze step ${st} ($1): ${src} -> ${dst}"
  mkdir -p "$(dirname "${dst}")"; rm -rf "${tmp}"
  cp -r "${src}" "${tmp}" || { log "copy failed"; return 1; }
  local f
  for f in "${src}"/*; do  # same files, same sizes
    [[ "$(stat -c %s "${f}")" == "$(stat -c %s "${tmp}/$(basename "${f}")" 2>/dev/null)" ]] || { log "size mismatch ${f}"; return 1; }
  done
  rm -rf "${dst}"; mv "${tmp}" "${dst}"
  cp "${RUN}/lingbotvla_cli.yaml" "${OUT}/lingbotvla_cli.yaml"
  ln -sfn "global_step_${st}" "${OUT}/checkpoints/current"
  echo "${st}" > "${E}/STEP"
  echo "$(date '+%F %T') step ${st}: $1" > "${E}/REASON"
  touch "${dst}/.ready"
  log "ready: ${OUT}/checkpoints/current/hf_ckpt (step ${st})"
}

[[ -f "${STATE}" ]] || date +%s > "${STATE}"
log "watching ${RUN} (job name ${JOB_NAME}, max steps ${MAX_STEPS}, stall ${STALL_MIN} min, freeze at ${FREEZE_AT:-never}$( \
    (( FREEZE_AT_S )) && date -d "@${FREEZE_AT_S}" '+ = %F %T %Z'), or when eval ${EVAL_DONE:-(none)} is done, \
${#ALL_TASKS[@]} tasks); old arrays: ${OLD_ARRAYS:-none}"
while true; do
  load_overrides
  if [[ -n "${OLD_ARRAYS:-}" ]]; then scancel -t PENDING ${OLD_ARRAYS} 2>/dev/null; fi
  if [[ ! -f "${OUT}/checkpoints/current/hf_ckpt/.ready" ]]; then
    if [[ -f "${E}/FREEZE_NOW" ]]; then
      step=$(newest_step); freeze "FREEZE_NOW (newest checkpoint ${step:-none})"
    elif (( FREEZE_AT_S && $(date +%s) >= FREEZE_AT_S )); then
      step=$(newest_step); freeze "time cutoff ${FREEZE_AT} (newest checkpoint ${step:-none})"
    elif [[ -n "${EVAL_DONE}" ]] && eval_done; then
      step=$(newest_step); freeze "eval ${EVAL_DONE} finished (newest checkpoint ${step:-none})"
    elif n=$(squeue -h -u "${USER}" -n "${JOB_NAME}" -t R 2>/dev/null | wc -l) && squeue -h -u "${USER}" >/dev/null 2>&1; then
      now=$(date +%s)
      (( n > 0 )) && echo "${now}" > "${STATE}"
      idle=$(( (now - $(cat "${STATE}")) / 60 ))
      step=$(newest_step); step=${step:-0}
      if (( step >= MAX_STEPS )); then freeze "training reached step ${step}"
      elif (( n == 0 && idle >= STALL_MIN )); then freeze "no ${JOB_NAME} job running for ${idle} min (newest checkpoint ${step})"
      fi
    fi
  elif [[ -z "${OLD_ARRAYS:-}" ]] || [[ -z "$(squeue -h -u "${USER}" -j "${OLD_ARRAYS// /,}" 2>/dev/null)" ]]; then
    log "DINO checkpoint frozen and no old eval array left: done"; break
  fi
  sleep "${TRIGGER_INTERVAL_S:-300}"
done
