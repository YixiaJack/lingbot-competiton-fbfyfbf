#!/usr/bin/env bash
# On the Mac: wait for the AMD run's step-STEP checkpoint, then have the cluster pull its hf_ckpt directly from AMD
# (scripts/hpc/pull_amd_ckpt.sh, detached on the login node that holds the forwarded agent socket), verified with
# SHA256 and marked .ready for the eval pool (phase q100_amd3k_3000 of scripts/hpc/eval_pool.sbatch).
#   nohup caffeinate -i bash scripts/amd_ckpt_to_cluster.sh >> ~/LingBotRoboTwin/logs/amd_ckpt_to_cluster.log 2>&1 &
# The Mac only coordinates (AMD -> Mac is 0.5 MB/s, AMD -> cluster ~3 MB/s with parallel streams); it must stay awake and
# online (caffeinate) because the cluster reaches AMD with the Mac's agent through the fwd2 ssh master (Duo-authenticated).
# If that master is gone the script logs it, posts a macOS notification and waits for
#   ssh -fNM -A -o ControlPath=~/.ssh/cm/fwd2-%r@%h:%p ${CLUSTER_SSH:-cluster}
# Overrides for tests: AMD_SRC, CLUSTER_DST, READY=0, YAML="".
set -uo pipefail
STEP="${STEP:-3000}"
CLUSTER_SCRATCH="${CLUSTER_SCRATCH:?set CLUSTER_SCRATCH=/scratch/<user> (cluster side)}"; CLUSTER_HOME="${CLUSTER_HOME:?set CLUSTER_HOME (cluster home)}"
AMD_HOST="${AMD_HOST:?set AMD_HOST=root@<amd-host>}"; AMD_PORT="${AMD_PORT:?set AMD_PORT}"
AMD_RUN=/root/robotwin-scratch/dino_port/runs/cont8k_v3_nodepth3k_amd
SRC="${AMD_SRC:-${AMD_RUN}/checkpoints/global_step_${STEP}/hf_ckpt}"
DST="${CLUSTER_DST:-${CLUSTER_SCRATCH}/lingbot/runs/amd_cont8k_v3_nodepth3k/checkpoints/global_step_${STEP}/hf_ckpt}"
# the policy server reads <ckpt>/../../../lingbotvla_cli.yaml; the AMD run's own yaml has AMD paths, so use the cluster
# yaml of the same architecture (22k+8k, no DINO); the AMD yaml is copied too, for the record
YAML="${YAML-${CLUSTER_SCRATCH}/lingbot/runs/strong_scratch22k_cont8k/lingbotvla_cli.yaml}"
READY="${READY:-1}"
FWD="${FWD:-$HOME/.ssh/cm/fwd2-${CLUSTER_USER:?set CLUSTER_USER}@${CLUSTER_LOGIN:?set CLUSTER_LOGIN}:22}"
SOCK="${CLUSTER_HOME}/.ssh/amd_agent.sock"
PULL=${CLUSTER_SCRATCH}/lingbot/LingBotRoboTwin/scripts/hpc/pull_amd_ckpt.sh
TLOG="$(dirname "${DST}").pull.log"

log() { echo "$(date '+%F %T') $*"; }
notify() { osascript -e "display notification \"$1\" with title \"AMD → 我的算力检查点\"" >/dev/null 2>&1 || true; }
amd() { ssh -o BatchMode=yes -o ConnectTimeout=20 -p "${AMD_PORT}" "${AMD_HOST}" "$@"; }
tf() { ssh -o ControlPath="${FWD}" -o BatchMode=yes -o ConnectTimeout=20 ${CLUSTER_SSH:-cluster} "$@"; }

ensure_link() {  # the fwd2 master is up and the Mac agent is forwarded to SOCK on its login node
  local warned=0
  until ssh -O check -o ControlPath="${FWD}" ${CLUSTER_SSH:-cluster} >/dev/null 2>&1; do
    (( warned )) || { log "fwd2 master is down: run  ssh -fNM -A -o ControlPath=~/.ssh/cm/fwd2-%r@%h:%p ${CLUSTER_SSH:-cluster}  (Duo)"
                      notify "集群连接断了，需要重新 Duo 认证"; warned=1; }
    sleep 120
  done
  if ! tf "SSH_AUTH_SOCK=${SOCK} ssh-add -l >/dev/null 2>&1"; then
    tf "rm -f ${SOCK}"; ssh -o ControlPath="${FWD}" -O forward -R "${SOCK}:${SSH_AUTH_SOCK}" ${CLUSTER_SSH:-cluster} >/dev/null 2>&1
    tf "SSH_AUTH_SOCK=${SOCK} ssh-add -l >/dev/null 2>&1" && log "agent forwarded to ${SOCK}" || { log "agent forward failed"; return 1; }
  fi
}

amd_complete() {  # 0 once SRC has its index, every shard and nothing written for 3 minutes
  amd bash -s -- "${SRC}" <<'EOF'
H=$1
[ -f "$H/model.safetensors.index.json" ] || exit 1
for s in $(python3 -c 'import json,sys; print(" ".join(sorted(set(json.load(open(sys.argv[1]))["weight_map"].values()))))' "$H/model.safetensors.index.json"); do
  [ -s "$H/$s" ] || exit 1
done
[ -z "$(find "$H" -newermt '3 minutes ago' -print -quit)" ]
EOF
}

log "waiting for ${AMD_HOST}:${SRC}"
until amd_complete; do sleep 300; done
log "checkpoint complete on AMD; sha256 there"
SUMS_LOCAL=$(mktemp); amd "cd ${SRC} && sha256sum \$(ls -A | grep -v '^\.')" > "${SUMS_LOCAL}" || { log "sha256 on AMD failed"; exit 1; }
log "$(wc -l < "${SUMS_LOCAL}") files, $(amd "du -sb ${SRC} | cut -f1") bytes"
until ensure_link; do sleep 60; done
tf "mkdir -p $(dirname "${DST}") && cat > ${DST}.sha256" < "${SUMS_LOCAL}"
amd "cat $(dirname "$(dirname "$(dirname "${SRC}")")")/../../configs/cont_amd_nodepth3k.yaml" 2>/dev/null \
  | tf "cat > $(dirname "$(dirname "$(dirname "${DST}")")")/lingbotvla_cli.amd.yaml" || true
tf "cd ${CLUSTER_SCRATCH}/lingbot && SSH_AUTH_SOCK=${SOCK} nohup setsid bash ${PULL} ${SRC} ${DST} ${DST}.sha256 '${YAML}' ${READY} \
    >> ${TLOG} 2>&1 < /dev/null & echo \$! > ${DST}.pid"
log "pull started on $(tf hostname) (pid $(tf "cat ${DST}.pid")), log ${TLOG}"
TOTAL=$(awk '{print $2}' "${SUMS_LOCAL}" | while read -r f; do amd "stat -c %s ${SRC}/${f}" < /dev/null; done | awk '{s+=$1} END{print s}')
python3 - "${DST}" "${TOTAL}" "${STEP}" <<'PY' || true  # progress board (~/progress_board, kind cluster = bytes on the cluster)
import json, os, sys
dst, total, step = sys.argv[1], int(sys.argv[2]), sys.argv[3]
p = os.path.expanduser("~/progress_board/tasks.json")
t = [x for x in json.load(open(p)) if x["id"] != "amd_ckpt_pull"]
t.append({"id": "amd_ckpt_pull", "name": f"AMD → 我的算力：续训第 {step} 步 hf_ckpt", "kind": "cluster", "path": dst + ".partial", "total": total})
json.dump(t, open(p, "w"), ensure_ascii=False, indent=2)
PY
notify "AMD 第 ${STEP} 步检查点开始传到我的算力"
while true; do
  sleep 120
  ensure_link || continue
  if tf "test -f ${DST}.DONE"; then
    log "done: $(tf "tail -1 ${TLOG}")"; notify "AMD 检查点已到我的算力并核对通过"
    python3 -c "import json,os; p=os.path.expanduser('~/progress_board/tasks.json'); t=json.load(open(p))
for x in t:
    if x['id']=='amd_ckpt_pull': x['path']='${DST}'; x['done']=True
json.dump(t, open(p,'w'), ensure_ascii=False, indent=2)" || true
    break
  fi
  if tf "test -f ${DST}.FAILED"; then log "FAILED: $(tf "tail -3 ${TLOG}")"; notify "AMD 检查点传输校验失败"; exit 1; fi
  tf "kill -0 \$(cat ${DST}.pid) 2>/dev/null" || { log "pull process gone without DONE/FAILED; see ${TLOG}"; notify "AMD 检查点传输进程意外退出"; exit 1; }
done
