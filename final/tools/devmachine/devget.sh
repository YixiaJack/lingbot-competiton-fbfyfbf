#!/bin/bash
# devget.sh REMOTE LOCAL [session]  -- download a file from the dev machine through a logged-in tmux pane
# (base64 + tmux pipe-pane, md5 checked). Holds tq.sh's pane lock so keepalive / tq calls never type into the stream.
R=$1; L=$2; S=${3:-devq}; DEV=<dev-hostname>
LK=/tmp/tq_${S}.lock; for i in $(seq 1 120); do mkdir "$LK" 2>/dev/null && break; sleep 0.5; done
[ -d "$LK" ] || { echo "pane busy"; exit 3; }
LOG=$(mktemp); trap 'tmux pipe-pane -t "$S"; rmdir "$LK" 2>/dev/null; rm -f "$LOG" "$LOG.body"' EXIT
last=$(tmux capture-pane -p -t "$S" | grep -v "^[[:space:]]*$" | tail -1)
case "$last" in *'$ '*|*'$') ;; *) echo "pane not at a shell prompt: ${last:0:60}"; exit 4;; esac
tmux send-keys -t "$S" -l "clear"; tmux send-keys -t "$S" Enter; sleep 1
tmux pipe-pane -t "$S" -o "cat > $LOG"; t0=$(date +%s)
tmux send-keys -t "$S" -l "if [ \"\$(hostname)\" = $DEV ]; then echo __DL\"\"B__; md5sum $R; base64 -w 76 $R; echo __DL\"\"E__; else echo __DL\"\"X__; fi"; tmux send-keys -t "$S" Enter
until perl -pe 's/\e\[[0-9;?]*[A-Za-z]//g; s/\r//g' "$LOG" 2>/dev/null | tail -c 400 | grep -q -E '^__DL(E|X)__$'; do
  sleep 2; [ $(( $(date +%s) - t0 )) -gt 3600 ] && { echo TIMEOUT; exit 1; }; done
perl -pe 's/\e\[[0-9;?]*[A-Za-z]//g; s/\r//g' "$LOG" | grep -q '^__DLX__$' && { echo "NOT ON THE DEV MACHINE"; exit 2; }
perl -pe 's/\e\[[0-9;?]*[A-Za-z]//g; s/\r//g' < "$LOG" | awk '/^__DLB__$/{f=1;next} /^__DLE__$/{f=0} f' > "$LOG.body"
md5r=$(head -1 "$LOG.body" | awk '{print $1}'); tail -n +2 "$LOG.body" | base64 -d > "$L"
md5l=$(md5 -q "$L"); dt=$(( $(date +%s) - t0 ))
[ "$md5r" = "$md5l" ] && echo "OK $L $(stat -f %z "$L") bytes in ${dt}s" || { echo "MD5 MISMATCH remote $md5r local $md5l"; exit 1; }
