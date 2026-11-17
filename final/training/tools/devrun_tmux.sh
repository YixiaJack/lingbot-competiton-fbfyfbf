#!/bin/bash
# usage: devrun_tmux.sh [timeout_s] 'command'  -- runs in tmux session dev, prints output between markers
T=60; if [[ $1 =~ ^[0-9]+$ ]]; then T=$1; shift; fi
tmux has-session -t ${DEVSESS:-dev} 2>/dev/null || { echo NO_SESSION; exit 2; }
id=$RANDOM$RANDOM
tmux send-keys -t ${DEVSESS:-dev} "echo __B${id}__; { $*; } 2>&1; echo; echo __E${id}__" Enter
end=$((SECONDS+T))
while (( SECONDS < end )); do
  sleep 1
  tmux capture-pane -p -J -t ${DEVSESS:-dev} -S -3000 | grep -q "^__E${id}__" && break
done
tmux capture-pane -p -J -t ${DEVSESS:-dev} -S -3000 | awk -v b="__B${id}__" -v e="__E${id}__" '$0==b{f=1;next} $0==e{f=0} f'
