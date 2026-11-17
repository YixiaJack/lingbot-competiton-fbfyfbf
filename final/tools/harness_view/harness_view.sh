#!/bin/bash
# harness_view.sh start|stop|status|open|lan|local  -- live window of what the 5080 tells the VLA (read-only on the 5080)
#   lan   : restart listening on the LAN (teammates open the printed link with the access code); local: back to this Mac only
D=$(cd "$(dirname "$0")" && pwd); PORT=${PORT:-8766}; PID=$D/server.pid; LOG=$D/server.log
running() { [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; }
case "$1" in
  start) running && { echo "already running (pid $(cat "$PID")) http://127.0.0.1:$PORT"; exit 0; }
         BIND=$(cat "$D/bind" 2>/dev/null || echo 127.0.0.1) PORT=$PORT nohup python3 "$D/server.py" > "$LOG" 2>&1 & echo $! > "$PID"; sleep 1
         running && echo "started (pid $(cat "$PID")) http://127.0.0.1:$PORT" || { echo "failed:"; tail -5 "$LOG"; exit 1; } ;;
  stop)  running && { kill "$(cat "$PID")"; rm -f "$PID"; echo stopped; } || echo "not running" ;;
  status) running && echo "running (pid $(cat "$PID")) http://127.0.0.1:$PORT" || echo "not running" ;;
  open)  "$0" start >/dev/null; open "http://127.0.0.1:$PORT" ;;
  lan)   echo 0.0.0.0 > "$D/bind"; "$0" stop >/dev/null; "$0" start >/dev/null
         IP=$(ipconfig getifaddr "$(route -n get <inference-host> 2>/dev/null | awk '/interface:/{print $2}')" 2>/dev/null)
         echo "LAN link for teammates: http://${IP:-<mac-ip>}:$PORT/?k=$(cat "$D/access_code")" ;;
  local) echo 127.0.0.1 > "$D/bind"; "$0" stop >/dev/null; "$0" start ;;
  *) echo "usage: $0 start|stop|status|open|lan|local"; exit 2 ;;
esac
