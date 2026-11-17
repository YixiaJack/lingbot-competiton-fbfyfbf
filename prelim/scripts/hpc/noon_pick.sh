#!/usr/bin/env bash
# 10/26 noon (Beijing) pick for the final 7-episode comparison with 22k+8k: the highest step of the 15k continuation
# (main run s22k8k_v3uni_15k or its B200 branch s22k8k_v3uni_15k_b200) that has a complete hf_ckpt is hard-linked to
# runs/v3uni_noon/checkpoints/pick/hf_ckpt (no copy; stays even if the run's cleanup removes the original), that run's
# lingbotvla_cli.yaml is put next to it (the policy server reads <ckpt>/../../../lingbotvla_cli.yaml), and
# <ckpt>/.ready is written last: the eval pool phase q100_v3uni_noon waits for it. Does nothing once .ready exists.
# Only checkpoints saved at a multiple of the member's save_steps have an hf_ckpt (the time-based saves do not).
#   bash scripts/hpc/noon_pick.sh          # DRY=1: only print the pick
set -uo pipefail
R="/scratch/${USER}/lingbot/runs"; OUT="${R}/v3uni_noon"; DST="${OUT}/checkpoints/pick/hf_ckpt"
NORM_SHA=92b37257234b37c8642894997847da496c6e6bb550c2968eced4439c8b6bed31  # clean-only norm stats
log() { echo "$(date '+%F %T') noon_pick: $*"; }
[[ -f "${DST}/.ready" ]] && { log "already picked: $(cat "${OUT}/PICK")"; exit 0; }
hf_complete() {  # hf_ckpt dir -> 0 if config, index and every shard named in the index are there, nothing written for 3 min
  local h=$1 f shards
  [[ -s "${h}/config.json" && -s "${h}/model.safetensors.index.json" ]] || return 1
  shards=$(python3 -c "import json,sys; print('\n'.join(sorted(set(json.load(open(sys.argv[1]))['weight_map'].values()))))" \
           "${h}/model.safetensors.index.json") || return 1
  for f in ${shards}; do [[ -s "${h}/${f}" ]] || return 1; done
  [[ -z "$(find "${h}" -newermt '3 minutes ago' -print -quit)" ]]
}
best=""; best_run=""
for run in s22k8k_v3uni_15k s22k8k_v3uni_15k_b200; do  # same step in both (a linked checkpoint): the main run's
  for st in $(ls "${R}/${run}/checkpoints" 2>/dev/null | grep -E '^global_step_[0-9]+$' | sed 's/global_step_//' | sort -rn); do
    hf_complete "${R}/${run}/checkpoints/global_step_${st}/hf_ckpt" || continue
    if [[ -z "${best}" ]] || (( st > best )); then best=${st}; best_run=${run}; fi
    break
  done
done
[[ -n "${best}" ]] || { log "no complete hf_ckpt in either run"; exit 1; }
SRC="${R}/${best_run}/checkpoints/global_step_${best}"
log "pick global_step_${best} of ${best_run}"
[[ -n "${DRY:-}" ]] && exit 0
ns=$(grep -oE "norm_stats_file: *[^ ]+" "${R}/${best_run}/lingbotvla_cli.yaml" | awk '{print $2}')
[[ "$(sha256sum "${ns}" | cut -d' ' -f1)" == "${NORM_SHA}" ]] || { log "norm stats ${ns} is not the clean-only file"; exit 1; }
mkdir -p "${OUT}/checkpoints/pick"
rm -rf "${DST}" "${DST}.partial"  # an earlier attempt that stopped before .ready (hard links only; the source stays)
cp -al "${SRC}/hf_ckpt" "${DST}.partial" && mv "${DST}.partial" "${DST}" || { log "hard link failed"; exit 1; }
for f in "${SRC}"/hf_ckpt/*; do
  [[ "$(stat -c %i "${f}")" == "$(stat -c %i "${DST}/${f##*/}")" ]] || { log "${f##*/} not linked"; exit 1; }
done
cp "${R}/${best_run}/lingbotvla_cli.yaml" "${OUT}/lingbotvla_cli.yaml"
echo "global_step_${best} ${best_run} $(date '+%F %T')" > "${OUT}/PICK"
touch "${DST}/.ready"
log "ready: ${DST} ($(ls "${DST}" | wc -l) files, norm stats ok)"
