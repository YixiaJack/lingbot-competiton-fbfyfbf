#!/usr/bin/env bash
# Worker-pool evaluation: a job keeps its GPU and pulls (setting, task) items from a shared list until none are left,
# instead of one array element per task (which gives the GPU back after ~10 min and waits in the queue again).
# Each GPU runs SLOTS items at once (default: 2 on >= 70 GB cards, 1 otherwise; one item = server + client, ~31 GB).
# Items are claimed with an atomic mkdir in <eval>/_claims/; a claim whose owner job is no longer running (or is this
# job after a preemption requeue) is taken over, and the per-episode state file (EVAL_STATE_FILE) resumes the item.
#   CKPT=.../hf_ckpt EVAL_NAME=q50x7_x EVAL_TEST_NUM=7 CONFIGS="demo_clean demo_randomized" [TASK_LIST=a,b] \
#     scripts/hpc/submit.sh scripts/hpc/eval_pool.sbatch --array=0-15 --constraint="a100|h100|l40s"
# PHASES="name|configs|test_num|tasks;name|configs|test_num|tasks;..." runs several pools one after the other (a slot
# moves to the next phase once it finds nothing left to claim in the current one; tasks empty = all 50), e.g. a short
# diagnostic first and then extending an earlier run: to extend <name> from 7 to 20 episodes per task, copy its
# directory without the *.result files; the per-episode state files make every item continue at episode 8.
# An item counts as done when its .result reports >= the phase's episode count, so phases can also raise the count on
# the same directory (e.g. 20 -> 40 -> ... -> 100): each item resumes from its state file at the next episode.
# A phase may name its own checkpoint as a 5th field, "name|configs|test_num|tasks|ckpt" (default: CKPT). Such a phase
# waits for <ckpt>/.ready (written once the checkpoint has been copied in and verified) and, from then on, comes first:
# a slot takes the first claimable item of the first ready phase every time it looks for work, so workers can be queued
# before the checkpoint exists, evaluate the other phases meanwhile and switch after their current item. When only a
# not-ready phase is left, the slot waits for it (polling every 10 min) instead of ending.
# A 6th field sets LINGBOT_DIR (the policy-server code) for that phase, e.g. the DINO dev checkout for DINO checkpoints:
# "name|configs|test_num|tasks|ckpt|lingbot_dir". The server finds the training config at <ckpt>/../../../
# lingbotvla_cli.yaml, so a checkpoint must sit at <run>/checkpoints/<step>/hf_ckpt next to <run>/lingbotvla_cli.yaml.
# Claims: a claim is taken over only when its owner is known not to run (a failed squeue is not taken as "not
# running"), and a takeover is re-read after 2 s so that two workers taking over at once cannot both keep it
# (10/24: 9 items were run by two workers at the same time, the second worker's log lines interleaved with the first's).
# Small requests start sooner on a full cluster: one item needs ~2 cores and <= ~33 GB RAM (sacct of earlier evals), so
#   SLOTS=1 ... eval_pool.sbatch --cpus-per-task=4 --mem=48G --time=02:00:00
# works; 5 min before the time limit the job requeues itself and its items continue from their state files.
# SHARED_SERVER=1 (10/27): one policy server per checkpoint for all slots of the job (src/shared_server.sh ->
# src/shared_policy_server.py, which keeps each client's inference state apart) instead of one per item: ~25 GB once +
# ~6 GB per slot, so SLOTS defaults to 3 on an 80 GB GPU and 2 on an L40S (46 GB). Two checkpoints' servers do not fit
# on an L40S: a job that may land on one should get the phases of one checkpoint only. The cluster cancels GPU jobs that
# run > 2 h below a utilization threshold
# (A100: 50%; cluster admin mails of 10/26); one item per GPU averaged 15-25%. The first slot that needs a checkpoint
# starts its server (flock), the others wait for it; a server that died or stopped answering /healthz is restarted.
#   SHARED_SERVER=1 PHASES=... scripts/hpc/submit.sh scripts/hpc/eval_pool.sbatch --array=0-7 \
#     --constraint="a100|h100|l40s" --gres=gpu:1 --cpus-per-task=8 --mem=80G --time=03:00:00 --job-name=lb_evalq
# Every item also runs under src/eval_one_task.sh's watchdog (an episode that stops printing for 10 min is retried;
# the same episode hanging twice counts as a failure).
#SBATCH --job-name=lb_pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/lingbot/logs/pool_%x_%A_%a.log
set -uo pipefail
cd "${WORK_DIR:-/scratch/${USER}/lingbot/LingBotRoboTwin}"
source scripts/hpc/env.sh
requeue_self() {  # USR1 (--signal) shortly before the time limit: back into the queue instead of ending
  echo "$(date +%T) time limit near: requeue ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}"
}
trap requeue_self USR1
: "${CKPT:?}"
PHASES="${PHASES:-${EVAL_NAME:?}|${CONFIGS:-demo_clean demo_randomized}|${EVAL_TEST_NUM:-100}|${TASK_LIST:-}}"
ME="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID:-0}"
GPU_MEM_MB=$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits)
if [[ "${SHARED_SERVER:-0}" == 1 ]]; then
  SLOTS="${SLOTS:-$(( GPU_MEM_MB >= 70000 ? 3 : 2 ))}"
else
  SLOTS="${SLOTS:-$(( GPU_MEM_MB >= 70000 ? 2 : 1 ))}"
fi
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-${WORK_DIR}/configs/nvidia_icd.json}"
mapfile -t ALL_TASKS < <(python3 -c "import json; print('\n'.join(json.load(open('configs/tasks50.json'))))")
echo "=== pool worker ${ME} on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) slots=${SLOTS} $(date)"
SRV_DIR="/tmp/lb_srv_${SLURM_JOB_ID:-$$}"
if [[ "${SHARED_SERVER:-0}" == 1 ]]; then
  echo "    shared policy server per checkpoint (${SRV_DIR})"
  mkdir -p "${SRV_DIR}"
  trap 'for f in "${SRV_DIR}"/*.pid; do [[ -f "${f}" ]] && kill "$(cat "${f}")" 2>/dev/null; done' EXIT
fi

server_port() {  # ckpt lingbot_dir -> port of this job's healthy shared server for the checkpoint (starts one if needed)
  local key; key=$(echo -n "$1|$2" | md5sum | cut -c1-12)
  (
    flock 9
    # /healthz is answered by the server's event loop, which a forward pass blocks; the first one also compiles
    # (minutes). So wait up to 10 min for the answer: only a server that is gone or truly stuck gets restarted.
    healthy() { curl -sf -m 600 "http://127.0.0.1:$1/healthz" > /dev/null; }
    if [[ -f "${SRV_DIR}/${key}.port" ]] && healthy "$(cat "${SRV_DIR}/${key}.port")"; then
      cat "${SRV_DIR}/${key}.port"; exit 0
    fi
    [[ -f "${SRV_DIR}/${key}.pid" ]] && kill "$(cat "${SRV_DIR}/${key}.pid")" 2>/dev/null && sleep 10
    rm -f "${SRV_DIR}/${key}.port"
    local port; port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')
    echo "$(date +%T) starting shared policy server for $1 on ${port}" >&2
    # 9>&-: the server must not inherit (and keep holding) the lock
    LINGBOT_DIR="$2" scripts/hpc/run_in_env.sh bash src/shared_server.sh "$1" "${port}" \
      > "${LOG_DIR}/pool_${SLURM_JOB_NAME}_${ME}_server_${key}.log" 2>&1 9>&- &
    echo $! > "${SRV_DIR}/${key}.pid"
    for _ in $(seq 1 180); do  # model load + compile warm-up
      healthy "${port}" && { echo "${port}" > "${SRV_DIR}/${key}.port"; echo "${port}"; exit 0; }
      kill -0 "$(cat "${SRV_DIR}/${key}.pid")" 2>/dev/null || { echo "shared server died, see its log" >&2; exit 1; }
      sleep 5
    done
    exit 1
  ) 9> "${SRV_DIR}/${key}.lock"
}
echo "    phases: ${PHASES}"

# The running jobs, read at most once per scan for work (run_one resets RJ_SCAN): each claim held by another worker needs
# them, and 10/25 slurmctld took 1-2 min per squeue, so a worker passing ~10 claimed items sat idle for 10-30 min.
# Array elements print as <job>_<index>. Returns non-zero when squeue fails or takes longer than 120 s.
running_jobs() {
  if [[ -z "${RJ_SCAN:-}" ]]; then
    RJ_SCAN=1; RJ=$(timeout 120 squeue -h -u "${USER}" -t R,CG -o "%i"); RJ_RC=$?
  fi
  return "${RJ_RC}"
}

claim() {  # item slot -> 0 if this slot now owns the item (owner = <job>:<slot>)
  local d="${CLAIMS}/$1" me="${ME}:$2"
  if mkdir "${d}" 2>/dev/null; then echo "${me}" > "${d}/owner"; return 0; fi
  local owner; owner=$(cat "${d}/owner" 2>/dev/null || true)
  if [[ -z "${owner}" ]]; then  # another worker created the claim and is about to write its name: not ours
    (( $(date +%s) - $(stat -c %Y "${d}") > 300 )) || return 1  # unless it stayed nameless for 5 min (worker died)
  fi
  local ojob="${owner%%:*}" oslot="${owner##*:}" rj
  # take over: this very slot after a preemption requeue, a slot this job no longer has, or a job that is not running
  if ! { [[ -z "${owner}" || "${owner}" == "${me}" ]] || { [[ "${ojob}" == "${ME}" ]] && (( oslot >= SLOTS )); }; }; then
    [[ "${ojob}" != "${ME}" ]] || return 1
    running_jobs || return 1                # squeue failed: cannot tell whether the owner runs, leave it
    rj="${RJ}"
    ! grep -qx "${ojob}" <<< "${rj}" || return 1
  fi
  echo "${me}" > "${d}/owner"
  sleep 2                                   # another worker taking it over at the same moment: the last writer keeps it
  [[ "$(cat "${d}/owner" 2>/dev/null)" == "${me}" ]] || return 1
  echo "take over $1 from ${owner:-?}"; return 0
}

slot() {  # slot index: run one item at a time, always the first claimable item of the first ready phase
  local s=$1
  IFS=';' read -r -a PH <<< "${PHASES}"
  declare -A fails=()
  echo "$(date +%T) slot ${s}: phases ${PHASES}"
  while true; do
    run_one "${s}" && continue
    (( WAITING )) || break                  # nothing left to claim anywhere
    sleep "${POOL_WAIT_S:-600}"              # only phases waiting for their checkpoint are left
  done
}

item_done() {  # cfg task -> 0 if <cfg>/<task>.result reports at least EVAL_TEST_NUM episodes ("successes episodes")
  local f="${ROOT}/$1/$2.result" n
  [[ -f "${f}" ]] || return 1
  n=$(awk '{print $2}' "${f}" 2>/dev/null)
  [[ -n "${n}" ]] && (( n >= EVAL_TEST_NUM ))
}

run_one() {  # slot -> 0 after running one item, 1 if nothing is claimable now (WAITING=1: a phase awaits its ckpt)
  local s=$1 phase name cfgs tn tl ck ld cfg task item tasks
  WAITING=0; RJ_SCAN=""
  for phase in "${PH[@]}"; do
    IFS='|' read -r name cfgs tn tl ck ld <<< "${phase}"
    if [[ -n "${ck}" && ! -f "${ck}/.ready" ]]; then WAITING=1; continue; fi
    ck="${ck:-${CKPT}}"
    ROOT="${PROJ}/eval/${name}"; CLAIMS="${ROOT}/_claims"; mkdir -p "${CLAIMS}"
    export EVAL_TEST_NUM="${tn}"
    if [[ -n "${tl}" ]]; then IFS=, read -r -a tasks <<< "${tl}"; else tasks=("${ALL_TASKS[@]}"); fi
    for cfg in ${cfgs}; do
      for task in "${tasks[@]}"; do
        item_done "${cfg}" "${task}" && continue
        item="${cfg}__${task}"
        (( ${fails[${name}/${item}]:-0} >= 2 )) && continue  # failed twice in this worker: leave it to the others
        claim "${item}" "${s}" || continue
        item_done "${cfg}" "${task}" && { rm -rf "${CLAIMS:?}/${item}"; continue; }
        echo "$(date +%T) slot ${s}: ${name} ${item} (${tn} episodes, ${ck}${ld:+, code ${ld}})"
        local sport=""
        if [[ "${SHARED_SERVER:-0}" == 1 ]] && ! sport=$(server_port "${ck}" "${ld:-${LINGBOT_DIR}}"); then
          echo "$(date +%T) slot ${s}: ${name} ${item} -> no shared server"
          fails[${name}/${item}]=$(( ${fails[${name}/${item}]:-0} + 1 )); rm -rf "${CLAIMS:?}/${item}"; return 0
        fi
        SHARED_POLICY_PORT="${sport}" LINGBOT_DIR="${ld:-${LINGBOT_DIR}}" scripts/hpc/run_in_env.sh bash src/eval_one_task.sh "${ck}" "${task}" "${cfg}" "${ROOT}/${cfg}" \
          > "${LOG_DIR}/pool_${SLURM_JOB_NAME}_${ME}_${name}_${item}.log" 2>&1
        echo "$(date +%T) slot ${s}: ${name} ${item} -> $(cat "${ROOT}/${cfg}/${task}.result" 2>/dev/null || echo unfinished)"
        item_done "${cfg}" "${task}" || fails[${name}/${item}]=$(( ${fails[${name}/${item}]:-0} + 1 ))
        rm -rf "${CLAIMS:?}/${item}"
        return 0
      done
    done
  done
  return 1
}

for ((s = 0; s < SLOTS; s++)); do
  slot "${s}" &
  sleep 60  # stagger model loading / torch.compile warm-up
done
until wait; do :; done  # a trapped signal interrupts wait (status > 128); keep waiting for the slots
echo "=== pool worker ${ME} done $(date)"
