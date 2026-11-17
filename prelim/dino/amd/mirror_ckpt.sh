#!/usr/bin/env bash
# Keep a copy of the newest checkpoint of an AMD run on the persistent volume: the container's own disk (/root) is lost
# when the instance is recreated (10/24 the whole 10/23 run went that way). Start it next to the training:
#   bash dino/amd/mirror_ckpt.sh [run dir]        # default runs/cont8k_v3_nodepth4k_amd; detaches, pid in <run>/mirror.pid
# Every 5 minutes: the newest complete checkpoint in <run>/checkpoints (extra_state, model/ and optimizer/ with their
# .metadata, hf_ckpt with every shard of its index; nothing written in it for 3 minutes) that is newer than the mirrored
# one replaces it in ${PERSIST}/<run name>/ (default /workspace/lingbot_ckpt). One checkpoint is ~75 GB and the volume
# 100 GB, so the old copy is deleted first, the new one is copied to global_step_N.partial and renamed when complete;
# if even then the volume is too small, only hf_ckpt + extra_state are copied (enough to evaluate, not to resume the
# optimizer). The checkpoints under /root are not touched. Exits after the training has ended and its last checkpoint is
# mirrored. Log: <run>/mirror.log.
# Resume after a lost container: copy ${PERSIST}/<run name>/global_step_N back to <run>/checkpoints/ and start
# dino/amd/train_amd.sh again (it resumes from the newest checkpoint in the run dir).
set -uo pipefail
RUN="$(realpath "${1:-runs/cont8k_v3_nodepth4k_amd}")"
PERSIST="${PERSIST:-/workspace/lingbot_ckpt}"
DEST="${PERSIST}/$(basename "${RUN}")"
if [[ "${MIRROR_CHILD:-0}" != 1 ]]; then
  mkdir -p "${RUN}" "${DEST}"
  MIRROR_CHILD=1 nohup setsid bash "$0" "${RUN}" >> "${RUN}/mirror.log" 2>&1 < /dev/null &
  echo $! > "${RUN}/mirror.pid"; echo "mirror started pid $(cat "${RUN}/mirror.pid"), log ${RUN}/mirror.log -> ${DEST}"
  exit 0
fi

log() { echo "$(date '+%F %T') $*"; }
complete() {  # checkpoint dir -> 0 if fully written
  local c=$1
  [[ -f "${c}/extra_state/extra_state_rank_0.pt" && -f "${c}/model/.metadata" && -f "${c}/optimizer/.metadata" \
     && -f "${c}/hf_ckpt/model.safetensors.index.json" ]] || return 1
  local s
  for s in $(python3 -c "import json,sys; print(' '.join(sorted(set(json.load(open(sys.argv[1]))['weight_map'].values()))))" \
             "${c}/hf_ckpt/model.safetensors.index.json"); do
    [[ -s "${c}/hf_ckpt/${s}" ]] || return 1
  done
  [[ -z "$(find "${c}" -newermt "${MIRROR_QUIET_MIN:-3} minutes ago" -print -quit)" ]]  # nothing written lately
}
mirrored() { ls -d "${DEST}"/global_step_* 2>/dev/null | grep -v '\.partial$' | sed 's/.*global_step_//' | sort -n | tail -1; }
training_alive() { [[ -f "${RUN}/train.pid" ]] && kill -0 "$(cat "${RUN}/train.pid")" 2>/dev/null; }

log "mirror ${RUN} -> ${DEST}"
final=0
while true; do
  alive=0; training_alive && alive=1
  newest=""
  for c in $(ls -d "${RUN}"/checkpoints/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -rn); do
    complete "${RUN}/checkpoints/global_step_${c}" && { newest=${c}; break; }
  done
  have=$(mirrored)
  if [[ -n "${newest}" && "${newest}" != "${have:-}" ]]; then
    src="${RUN}/checkpoints/global_step_${newest}"
    need=$(du -sb "${src}" | cut -f1)
    for old in "${DEST}"/global_step_*; do [[ -e "${old}" ]] && { log "remove old copy ${old}"; rm -rf "${old}"; }; done
    avail=$(df -B1 --output=avail "${PERSIST}" | tail -1)
    parts=(extra_state hf_ckpt model optimizer)
    if (( need + 2 * 1024**3 > avail )); then
      parts=(extra_state hf_ckpt)
      log "WARNING: ${PERSIST} has $((avail / 1024**3)) GB free, step ${newest} needs $((need / 1024**3)) GB: copying hf_ckpt + extra_state only"
    fi
    log "copy step ${newest} (${parts[*]})"
    mkdir -p "${DEST}/global_step_${newest}.partial"
    ok=1
    for p in "${parts[@]}"; do
      nice -n 19 ionice -c3 cp -r "${src}/${p}" "${DEST}/global_step_${newest}.partial/" || ok=0
    done
    if (( ok )); then
      mv "${DEST}/global_step_${newest}.partial" "${DEST}/global_step_${newest}"
      log "mirrored step ${newest}: $(du -sh "${DEST}/global_step_${newest}" | cut -f1)"
    else
      log "copy of step ${newest} failed; retrying next round"; rm -rf "${DEST}/global_step_${newest}.partial"
    fi
    continue
  fi
  if (( ! alive )); then  # one more round after the training ended: its last checkpoint may still be settling
    (( final )) && { log "training not running, newest checkpoint (${newest:-none}) mirrored: done"; break; }
    final=1
  fi
  sleep "${MIRROR_INTERVAL:-300}"
done
