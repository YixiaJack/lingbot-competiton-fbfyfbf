#!/usr/bin/env bash
# Before a member of a run starts training (train.sbatch PRE_TRAIN_HOOK), take over the other line's checkpoint if it is
# further: the main run s22k8k_v3uni_15k and its B200 branch s22k8k_v3uni_15k_b200 both call this at every start, each
# with the other one as source, so whichever line has trained further is the one that continues.
#   bash scripts/hpc/branch_seed.sh <source run dir> <this run dir>       # DRY=1: only print what it would do
# A checkpoint counts when it is complete: model/ and optimizer/ .metadata, every extra_state_rank_*.pt non-empty, nothing
# written in it for 3 minutes. If the source's newest complete checkpoint has a higher step than this run's newest
# complete one (or this run has none), it is hard-linked (cp -al: no copy, no extra space; the trainer only reads it and
# saves new checkpoints in new directories; each run's cleanup only removes its own links) into <this run>/checkpoints/,
# and the trainer resumes there: weights, optimizer, lr schedule, step. Otherwise nothing is done and the run resumes from
# its own checkpoints. Exit status non-zero only when this run has no checkpoint at all and none could be linked: the
# caller must not start training from scratch then (a branch's first start before the source has saved anything).
set -uo pipefail
SRC="${1:?source run dir}"; DST="${2:?this run dir}"
log() { echo "$(date '+%F %T') branch_seed: $*"; }
complete() {  # checkpoint dir -> 0 if fully written
  local c=$1 r n=0
  [[ -f "${c}/model/.metadata" && -f "${c}/optimizer/.metadata" ]] || return 1
  for r in "${c}"/extra_state/extra_state_rank_*.pt; do [[ -s "${r}" ]] || return 1; n=$((n + 1)); done
  (( n > 0 )) || return 1
  [[ -z "$(find "${c}" -newermt '3 minutes ago' -print -quit)" ]]
}
newest_complete() {  # run dir -> step of its newest complete checkpoint (empty if none)
  local st
  for st in $(ls "$1/checkpoints" 2>/dev/null | grep -E '^global_step_[0-9]+$' | sed 's/global_step_//' | sort -rn); do
    complete "$1/checkpoints/global_step_${st}" && { echo "${st}"; return; }
  done
}
own=$(newest_complete "${DST}")
any_own=$(ls "${DST}/checkpoints" 2>/dev/null | grep -cE '^global_step_[0-9]+$')
src=$(newest_complete "${SRC}")
if [[ -z "${src}" ]]; then
  (( any_own > 0 )) && { log "nothing to take from ${SRC}; resume from own checkpoints (newest complete ${own:-none})"; exit 0; }
  log "${DST} has no checkpoint and ${SRC} has no complete one: cannot start"; exit 1
fi
if [[ -n "${own}" ]] && (( own >= src )); then
  log "own global_step_${own} >= ${SRC##*/} global_step_${src}: resume from own"; exit 0
fi
if [[ -e "${DST}/checkpoints/global_step_${src}" ]]; then  # an incomplete own checkpoint with that step: leave it alone
  (( any_own > 0 )) && { log "own incomplete global_step_${src} exists; resume from own checkpoints"; exit 0; }
fi
log "take ${SRC}/checkpoints/global_step_${src} (own newest complete: ${own:-none})"
[[ -n "${DRY:-}" ]] && exit 0
mkdir -p "${DST}/checkpoints"
if cp -al "${SRC}/checkpoints/global_step_${src}" "${DST}/checkpoints/global_step_${src}.partial" \
   && mv "${DST}/checkpoints/global_step_${src}.partial" "${DST}/checkpoints/global_step_${src}"; then
  log "linked global_step_${src} ($(find "${DST}/checkpoints/global_step_${src}" -type f | wc -l) files)"
else
  rm -rf "${DST}/checkpoints/global_step_${src}.partial"
  (( any_own > 0 )) && { log "hard link failed; resume from own checkpoints"; exit 0; }
  log "hard link failed and no own checkpoint: cannot start"; exit 1
fi
