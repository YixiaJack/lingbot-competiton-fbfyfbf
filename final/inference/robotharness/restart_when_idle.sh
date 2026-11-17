#!/bin/bash
# 5080: restart lingbot_infer (same checkpoint, RTC) via ~/rtc/switch.sh once the last takeover event is "stop" and
# at least 90 s old, so the bridge whitelist with proprio2 is loaded. Gives up after 60 min. Log: ~/robotharness/restart_1115.log
R=$HOME/workspace/lingbot/inference
for i in $(seq 1 360); do
  age=$(tail -1 "$R/logs/takeover_events.jsonl" | python3 -c 'import sys, json, time; d = json.loads(sys.stdin.read()); print(int(time.time() - d["wall_time"]) if d["event"] == "stop" else -1)')
  if [ "${age:--1}" -ge ${1:-90} ]; then
    echo "$(date +%T) idle ${age}s -> switch.sh rtc"
    out=$(bash "$HOME/rtc/switch.sh" rtc 2>&1); rc=$?
    echo "$out"
    if echo "$out" | grep -q "a session is running"; then echo "$(date +%T) session started first, waiting again"; sleep 10; continue; fi
    echo "SWITCH_EXIT $rc"; exit 0
  fi
  sleep 10
done
echo "GAVE_UP no 90 s idle gap in 60 min"
