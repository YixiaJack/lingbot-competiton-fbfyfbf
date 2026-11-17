#!/bin/bash
# Training progress for https://lingbot-board.vercel.app/train — keeps train_board.py running on this computer.
#   bash train_board.sh start|stop|status|log
# caffeinate -i keeps the Mac from idle-sleeping while the collector runs (closing the lid still sleeps it).
D=$(cd "$(dirname "$0")" && pwd); S=$HOME/.cache/train_board; mkdir -p "$S"; PID=$S/pid; LOG=$S/train_board.log
running(){ [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; }
case "$1" in
  start) if running; then echo "already running (pid $(cat "$PID"))"; exit 0; fi
         nohup caffeinate -i python3 -u "$D/train_board.py" --interval "${INTERVAL:-120}" >> "$LOG" 2>&1 &
         echo $! > "$PID"; sleep 1; echo "started pid $(cat "$PID"), log $LOG" ;;
  stop)  if running; then p=$(cat "$PID"); kill $(pgrep -P "$p") "$p" 2>/dev/null; rm -f "$PID"; echo stopped; else echo "not running"; fi ;;
  status) if running; then echo "running pid $(cat "$PID")"; else echo "not running"; fi; tail -3 "$LOG" 2>/dev/null ;;
  log)   tail -n "${2:-30}" "$LOG" ;;
  *) sed -n 2,4p "$0" ;;
esac
