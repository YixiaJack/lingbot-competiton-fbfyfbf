#!/usr/bin/env bash
# On the Mac until END (10/26 12:00 Beijing): every 5 min run scripts/hpc/gpu_cap_check.py on cluster (it cancels a few eval
# pool workers when a faster member of the s22k8k_v3uni_15k training group is held back by the per-user GPU cap) and
# post a macOS notification whenever it cancels something.
#   nohup caffeinate -i bash scripts/gpu_cap_watch.sh >> logs/gpu_cap_watch.log 2>&1 < /dev/null & echo $! > logs/gpu_cap_watch.pid
set -uo pipefail
cd "$(dirname "$0")/.."
END_S=$(date -j -f '%Y-%m-%d %H:%M %z' "${END:-2026-10-26 12:00 +0800}" +%s)
log() { echo "$(date '+%F %T') $*"; }
log "watch until $(date -r "${END_S}" '+%F %T')"
while (( $(date +%s) < END_S )); do
  out=$(timeout 200 ssh -o BatchMode=yes -o ConnectTimeout=20 ${CLUSTER_SSH:-cluster} 'python3 -' < scripts/hpc/gpu_cap_check.py 2>&1) || out="check failed: ${out}"
  log "${out}"
  if [[ "${out}" == *"cancelled eval workers"* ]]; then
    osascript -e "display notification \"为让训练排上，停了几个评测 worker（详见 logs/gpu_cap_watch.log）\" with title \"GPU 上限\"" >/dev/null 2>&1 || true
  fi
  sleep 300
done
log "watch ends"
