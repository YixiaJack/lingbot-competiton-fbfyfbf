#!/bin/bash
# usage: tq.sh [-s session] [-t timeout] 'command'  -> runs in the logged-in tmux pane on the dev machine, prints its output.
# Guard: the command only runs if the pane's shell is on the dev machine (hostname check), so a pane whose ssh session has
# dropped back to the local shell never executes it (11/14 19:00: devq had dropped after ~20 min idle).
S=devq; T=120
while getopts s:t: o; do case $o in s) S=$OPTARG;; t) T=$OPTARG;; esac; done; shift $((OPTIND-1))
DEV=<dev-hostname>
# one command at a time per pane (the keepalive loop uses this script too)
L=/tmp/tq_${S}.lock; got=0; for ((w=0; w<T*2; w++)); do mkdir "$L" 2>/dev/null && { got=1; break; }; sleep 0.5; done
[ $got = 1 ] || { echo "[tq: busy]"; exit 3; }   # never type into a pane another tq call is using
trap 'rmdir "$L" 2>/dev/null' EXIT
id=$RANDOM$RANDOM; out=$(mktemp)
last=$(tmux capture-pane -p -t "$S" | grep -v "^[[:space:]]*$" | tail -1)
case "$last" in *"\$ "*|*"\$"|*"# "*|*"TB> "*) ;; *) echo "[tq: pane is not at a shell prompt (last line: ${last:0:60}); nothing sent]"; exit 4;; esac
tmux send-keys -t "$S" -l -- "clear; echo B${id}B; if [ \"\$(hostname)\" = $DEV ]; then ( $* ) 2>&1; else echo 'tq: NOT ON THE DEV MACHINE, command not run'; fi; echo E${id}E" ; tmux send-keys -t "$S" Enter
for ((i=0; i<T*2; i++)); do
  tmux capture-pane -p -J -t "$S" -S -3000 > "$out"
  grep -q "^E${id}E" "$out" && break; sleep 0.5
done
awk -v b="B${id}B" -v e="E${id}E" '$0==b{f=1;next} $0==e{f=0} f' "$out"; grep -q "^E${id}E" "$out" || echo "[tq: timeout after ${T}s]"
rm -f "$out"
