#!/usr/bin/env bash
# On the Mac, until the AMD access ends: follow the AMD 3k -> 5k continuation (dino/amd/chain_cont.sh) and the evaluation
# after it (eval_amd/after_train_eval.sh), post a macOS notification for each milestone or error, and copy the evaluation
# results (eval_runs/: launcher logs, per-task logs, .result, state; no tmp/) to logs/amd_eval_runs/ every 10 minutes,
# so they survive the end of the AMD access.
#   nohup caffeinate -i bash scripts/amd_followup.sh >> logs/amd_followup.log 2>&1 < /dev/null &
set -uo pipefail
AMD_HOST="${AMD_HOST:?set AMD_HOST=root@<amd-host>}"; AMD_PORT="${AMD_PORT:?set AMD_PORT}"
PKG=/root/robotwin-scratch/dino_port
RUN="${PKG}/runs/cont8k_v3_nodepth3k_amd"
DEST="${HOME}/LingBotRoboTwin/logs/amd_eval_runs"
END_S=$(date -j -f '%Y-%m-%d %H:%M %z' "${END:-2026-10-26 01:30 +0800}" +%s)
SEEN="${HOME}/LingBotRoboTwin/logs/amd_followup.seen"  # milestones already notified
mkdir -p "${DEST}"; touch "${SEEN}"

log() { echo "$(date '+%F %T') $*"; }
notify() { osascript -e "display notification \"$1\" with title \"AMD 续训 / 评测\"" >/dev/null 2>&1 || true; }
amd() { ssh -o BatchMode=yes -o ConnectTimeout=20 -p "${AMD_PORT}" "${AMD_HOST}" "$@"; }
once() {  # key message -> notify and log the first time key is seen
  grep -qxF "$1" "${SEEN}" && return 0
  echo "$1" >> "${SEEN}"; log "$2"; notify "$2"
}
summary() {  # results copied so far: per run and setting, tasks finished and successes
  local r c
  for r in "${DEST}"/*/; do
    [[ -d "${r}demo_clean" || -d "${r}demo_randomized" ]] || continue
    for c in demo_clean demo_randomized; do
      ls "${r}${c}"/*.result >/dev/null 2>&1 || continue
      cat "${r}${c}"/*.result | awk -v n="$(basename "${r}")/${c}" '{s += $1; t += $2; k++} END {printf "%s: %d 项 %d/%d = %.1f%%  ", n, k, s, t, 100 * s / t}'
    done
  done
}

log "follow ${RUN} until $(date -r "${END_S}" '+%F %T')"
round=0
while (( $(date +%s) < END_S )); do
  st=$(amd "cat ${RUN}/chain.log 2>/dev/null; echo ===; cat ${RUN}/after_eval.log 2>/dev/null; echo ===; \
            tail -c 3000 \$(ls -t ${PKG}/runs/logs/cont_amd_*.log | head -1) | tr '\r' '\n' | grep -oE 'Step: [0-9]+/[0-9]+ \[[^]]*\]' | tail -1") \
    || { log "AMD not reachable"; sleep 300; continue; }
  chain=$(awk '/^===$/{n++; next} n == 0' <<< "${st}")
  after=$(awk '/^===$/{n++; next} n == 1' <<< "${st}")
  prog=$(awk '/^===$/{n++; next} n == 2' <<< "${st}")
  grep -q "ERROR" <<< "${chain}" && once chain_error "5k 续训没有启动：$(grep ERROR <<< "${chain}" | tail -1 | cut -c21-120)"
  grep -q "chain done" <<< "${chain}" && once chain_done "5k 续训已启动：$(grep -oE 'Step [0-9]+ VLA_Loss [0-9.]+ LR [0-9.e-]+' <<< "${chain}" | tail -1)"
  grep -q "WARNING" <<< "${chain}" && once chain_warn "5k 续训警告：$(grep WARNING <<< "${chain}" | tail -1 | cut -c21-120)"
  grep -q "training over" <<< "${after}" && once train_over "AMD 训练结束：$(grep 'training over' <<< "${after}" | cut -c21-)"
  grep -q "ERROR" <<< "${after}" && once eval_error "AMD 评测没有启动：$(grep ERROR <<< "${after}" | tail -1 | cut -c21-120)"
  grep -q "episodes, " <<< "${after}" && once eval_start "AMD 评测已开始：$(grep 'episodes, ' <<< "${after}" | head -1 | cut -c21-90)"
  while read -r l; do [[ -n "${l}" ]] && once "pass:${l:0:60}" "AMD 评测一轮结束：$(cut -c21-200 <<< "${l}")"; done <<< "$(grep 'pass ended' <<< "${after}")"
  if grep -q "episodes, " <<< "${after}" && (( round % 2 == 0 )); then
    rsync -a --timeout=120 -e "ssh -o BatchMode=yes -o ConnectTimeout=20 -p ${AMD_PORT}" --exclude='tmp/' \
      --include='*/' --include='*.log' --include='*.result' --include='*.json' --exclude='*' \
      "${AMD_HOST}:${PKG}/eval_runs/" "${DEST}/" && log "results: $(summary)" || log "rsync failed"
  fi
  (( round % 12 == 0 )) && log "training ${prog:-?}"
  grep -q "after_train_eval done" <<< "${after}" && { once eval_done "AMD 评测全部结束：$(summary)"; break; }
  round=$((round + 1)); sleep 300
done
rsync -a --timeout=120 -e "ssh -o BatchMode=yes -o ConnectTimeout=20 -p ${AMD_PORT}" --exclude='tmp/' \
  --include='*/' --include='*.log' --include='*.result' --include='*.json' --exclude='*' \
  "${AMD_HOST}:${PKG}/eval_runs/" "${DEST}/" 2>/dev/null && log "final results: $(summary)"
log "follow-up ends"
