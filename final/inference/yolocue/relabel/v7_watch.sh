#!/bin/bash
# Pause the YOLO trainer (whole process group) while lingbot-infer runs a session; resume when the session stops.
# usage: v7_watch.sh PGID   (the trainer was started with setsid, so PGID = its pid)
L=/home/guest/workspace/lingbot/inference/logs/takeover_events.jsonl
PG=$1; st=run
while kill -0 "$PG" 2>/dev/null; do
  ev=$(tail -n 1 "$L" | grep -o '"event": "[a-z]*"' | cut -d'"' -f4)
  if [ "$ev" != stop ] && [ "$st" = run ]; then kill -STOP -- "-$PG"; st=paused; echo "$(date '+%F %T') paused (robot session: $ev)"; fi
  if [ "$ev" = stop ] && [ "$st" = paused ]; then kill -CONT -- "-$PG"; st=run; echo "$(date '+%F %T') resumed"; fi
  sleep 0.5
done
echo "$(date '+%F %T') trainer gone, watcher exits"
