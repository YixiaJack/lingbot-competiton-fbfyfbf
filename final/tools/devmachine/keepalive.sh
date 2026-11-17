#!/bin/bash
# keepalive.sh [session] — the bastion closes a session after 600 s idle; run a no-op in the pane every 240 s.
# Stops by itself when the pane is no longer on the dev machine (never retries a login).
S=${1:-devq}; D=$(cd "$(dirname "$0")" && pwd)
while true; do
  out=$("$D/tq.sh" -s "$S" -t 30 'echo KEEPALIVE_OK' 2>&1)
  echo "$out" | grep -q -E 'KEEPALIVE_OK|\[tq: busy\]|not at a shell prompt' || { echo "$(date '+%H:%M:%S') keepalive: pane $S is not on the dev machine, stopping"; exit 1; }
  sleep 240
done
