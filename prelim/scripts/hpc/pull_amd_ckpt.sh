#!/usr/bin/env bash
# Pull one hf_ckpt from the AMD server to the cluster, verify it and hand it to the eval pool. Runs on the cluster login node that
# holds the forwarded agent socket (AMD accepts only the Mac's key; scripts/amd_ckpt_to_cluster.sh starts this detached).
#   SSH_AUTH_SOCK=$HOME/.ssh/amd_agent.sock bash pull_amd_ckpt.sh <amd hf_ckpt dir> <cluster dest hf_ckpt dir> <sha256 list>
#       [<yaml for the policy server, copied to dest/../../../lingbotvla_cli.yaml>] [ready=1]
# One rsync per file, all in parallel, each restarted every RESTART_S seconds (long-lived streams to the AMD box slow
# down to ~100 KB/s; a new connection is fast again), resuming with --append-verify. Then sha256 of every file against
# the list made on AMD; on success dest.partial -> dest, the yaml next to the run, and dest/.ready (eval pool gate).
# Writes <dest>.DONE or <dest>.FAILED.
set -uo pipefail
SRC=$1; DST=$2; SUMS=$3; YAML=${4:-}; READY=${5:-1}
AMD="${AMD:?set AMD=root@<amd-host>}"; PORT="${AMD_PORT:?set AMD_PORT}"; RESTART_S="${RESTART_S:-600}"
E="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p ${PORT}"
log() { echo "$(date '+%F %T') $*"; }
rm -f "${DST}.DONE" "${DST}.FAILED"
mkdir -p "${DST}.partial"
mapfile -t FILES < <(awk '{print $2}' "${SUMS}")
log "pull ${#FILES[@]} files ${AMD}:${SRC} -> ${DST}.partial"

pull() {
  local f=$1 n=0
  until timeout "${RESTART_S}" rsync -a --append-verify --timeout=120 -e "${E}" "${AMD}:${SRC}/${f}" "${DST}.partial/"; do
    n=$((n + 1)); sleep 5
    (( n % 6 == 0 )) && log "${f}: ${n} restarts, $(stat -c %s "${DST}.partial/${f}" 2>/dev/null || echo 0) bytes"
  done
  log "${f}: done ($(stat -c %s "${DST}.partial/${f}") bytes, ${n} restarts)"
}
for f in "${FILES[@]}"; do pull "${f}" & done
wait

log "sha256 check"
if (cd "${DST}.partial" && sha256sum -c --quiet "${SUMS}"); then
  rm -rf "${DST}"; mv "${DST}.partial" "${DST}"
  [[ -n "${YAML}" ]] && cp "${YAML}" "$(dirname "$(dirname "$(dirname "${DST}")")")/lingbotvla_cli.yaml"
  [[ "${READY}" == 1 ]] && touch "${DST}/.ready"
  log "verified: ${DST}$( [[ "${READY}" == 1 ]] && echo ' (.ready written)')"
  touch "${DST}.DONE"
else
  log "sha256 mismatch"; touch "${DST}.FAILED"
fi
