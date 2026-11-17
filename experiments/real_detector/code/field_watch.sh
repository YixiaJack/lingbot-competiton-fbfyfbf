#!/usr/bin/env bash
# Keep the board's field page (https://lingbot-board.vercel.app/field) up to date by itself: every EVERY seconds pull
# what is new from the machines that hold the LeRobot datasets (rsync over ssh, or a local path), and when anything
# changed rebuild and publish (field_sync.sh). Only meta/, data/ and the head camera's videos are copied.
#
#   bash field_watch.sh SOURCES [--every 180] [--once]
#
# SOURCES: one dataset per line, "location source collector" (# comments allowed), e.g.
#   dev:/mnt/forge/datasets/private/<id>/snapshots/<snap>/v1   demo     张三
#   dev:/mnt/forge/datasets/public/<id>/snapshots/<snap>/v1    demo     官方
#   ~/field_ds/local_copy                                      dagger   李四
# location: host:/path (any ssh host alias, key login, no password prompt) or a local directory. The file is re-read
# every round, so lines can be added while it runs. Mirrors live in FIELD_MIRROR (default ~/.cache/field_dev/mirror).
# Log: FIELD_MIRROR/watch.log. FIELD_CAMERA overrides the head camera (default: the video key with high/top/head).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SOURCES=${1:?usage: field_watch.sh SOURCES [--every SECONDS] [--once]}; shift
EVERY=180; ONCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in --every) EVERY="$2"; shift 2 ;; --once) ONCE=1; shift ;; *) echo "unknown option $1"; exit 2 ;; esac
done
MIRROR=${FIELD_MIRROR:-$HOME/.cache/field_dev/mirror}; mkdir -p "$MIRROR"
LOG="$MIRROR/watch.log"; STAMP="$MIRROR/.published_fingerprint"
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=20"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG" >&2; }   # stderr: functions return values on stdout

head_cam() {  # dataset dir -> head camera video key
  python3 - "$1" "${FIELD_CAMERA:-}" <<'PY'
import json, sys
info = json.load(open(sys.argv[1] + '/meta/info.json'))
if sys.argv[2]: print(sys.argv[2]); sys.exit()
cams = [k for k, v in info['features'].items() if v.get('dtype') == 'video']
print(next((k for k in cams if any(w in k for w in ('high', 'top', 'head'))), cams[0] if cams else ''))
PY
}

pull() {  # location -> local mirror dir (prints it); 1 if the pull failed
  local loc="$1" dst
  if [[ "$loc" != *:* || -d "$loc" ]]; then echo "$loc"; return 0; fi
  dst="$MIRROR/$(echo "$loc" | tr -c 'A-Za-z0-9._-' '_')"
  mkdir -p "$dst"
  rsync -az -e "ssh $SSH_OPTS" "$loc/meta/" "$dst/meta/" < /dev/null >> "$LOG" 2>&1 || return 1
  rsync -az -e "ssh $SSH_OPTS" "$loc/data/" "$dst/data/" < /dev/null >> "$LOG" 2>&1 || return 1
  local cam; cam=$(head_cam "$dst") || return 1
  [[ -n "$cam" ]] || { log "no video key in $loc"; return 1; }
  mkdir -p "$dst/videos/$cam"
  rsync -az -e "ssh $SSH_OPTS" "$loc/videos/$cam/" "$dst/videos/$cam/" < /dev/null >> "$LOG" 2>&1 || return 1
  echo "$dst"
}

fingerprint() {  # dataset dirs -> one line that changes when any dataset gains or changes episodes
  python3 - "$@" <<'PY'
import hashlib, json, os, sys
h = hashlib.sha1()
for d in sys.argv[1:]:
    info = json.load(open(os.path.join(d, 'meta/info.json')))
    h.update(f"{d}|{info.get('total_episodes')}|{info.get('total_frames')}".encode())
    for root, _, files in sorted(os.walk(os.path.join(d, 'meta/episodes'))):
        for f in sorted(files):
            st = os.stat(os.path.join(root, f)); h.update(f"{f}:{st.st_size}:{int(st.st_mtime)}".encode())
print(h.hexdigest())
PY
}

round_once() {
  local specs=() dirs=() n=0 loc src who dir
  while read -r loc src who _ <&3; do   # fd 3: ssh inside the loop must not eat the file
    [[ -z "${loc:-}" || "$loc" == \#* ]] && continue
    n=$((n + 1))
    if dir=$(pull "$loc"); then
      [[ -f "$dir/meta/info.json" ]] || { log "skip $loc: no meta/info.json yet"; continue; }
      specs+=("$dir:${src:-demo}:${who:-}"); dirs+=("$dir")
    else
      log "pull failed: $loc (kept the last mirror if any)"
      d="$MIRROR/$(echo "$loc" | tr -c 'A-Za-z0-9._-' '_')"
      [[ -f "$d/meta/info.json" ]] && { specs+=("$d:${src:-demo}:${who:-}"); dirs+=("$d"); }
    fi
  done 3< "$SOURCES"
  [[ ${#dirs[@]} -gt 0 ]] || { log "no dataset available yet ($n lines in $SOURCES)"; return 0; }
  local fp; fp=$(fingerprint "${dirs[@]}") || { log "fingerprint failed"; return 0; }
  if [[ -f "$STAMP" && "$(cat "$STAMP")" == "$fp" ]]; then log "no change (${#dirs[@]} datasets)"; return 0; fi
  local title; title="自动同步 $(TZ=Asia/Shanghai date '+%m/%d %H:%M')（北京时间）· ${#dirs[@]} 个数据集"
  if bash "$HERE/field_sync.sh" --title "$title" "${specs[@]}" < /dev/null >> "$LOG" 2>&1; then
    echo "$fp" > "$STAMP"; log "published: $title"
  else
    log "publish failed, see $LOG"
  fi
}

log "watch start: $SOURCES every ${EVERY}s"
while true; do
  round_once
  [[ $ONCE == 1 ]] && break
  sleep "$EVERY"
done
