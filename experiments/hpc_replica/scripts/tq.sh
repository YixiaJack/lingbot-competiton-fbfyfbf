#!/bin/bash
# usage: tq.sh [-s session] [-t timeout] 'command'  -> runs in the logged-in tmux pane, prints its output
S=devq; T=120
while getopts s:t: o; do case $o in s) S=$OPTARG;; t) T=$OPTARG;; esac; done; shift $((OPTIND-1))
id=$RANDOM$RANDOM; out=$(mktemp)
tmux send-keys -t "$S" -l -- "clear; echo B${id}B; { $*; } 2>&1; echo E${id}E" ; tmux send-keys -t "$S" Enter
for ((i=0; i<T*2; i++)); do
  tmux capture-pane -p -J -t "$S" -S -3000 > "$out"
  grep -q "^E${id}E" "$out" && break; sleep 0.5
done
awk -v b="B${id}B" -v e="E${id}E" '$0==b{f=1;next} $0==e{f=0} f' "$out"; grep -q "^E${id}E" "$out" || echo "[tq: timeout after ${T}s]"
rm -f "$out"
