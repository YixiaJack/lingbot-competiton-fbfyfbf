#!/bin/bash
# switch.sh sync|rtc [hf_ckpt] [--trained] [--dry-run]   (5080)
# Switch lingbot_infer between the official synchronous scheduler (config/inference.sync.yaml) and async + RTC
# (config/inference.rtc.yaml: use_length 44, refill_threshold 20), then restart the idle service and reload the policy.
#   hf_ckpt    checkpoint to load (default: the one loaded now, from the newest service log)
#   --trained  mark hf_ckpt as RTC-trained (append it to ~/rtc/allow.txt), so the prefix is used. Unmarked checkpoints
#              run async WITHOUT prefix and the service log warns.
# Refuses while a session runs (the last takeover event must be "stop"). Restart = kill -9 -> systemd (no sudo).
set -euo pipefail
R=$HOME/workspace/lingbot/inference
MODE=${1:-}; shift || true
MODEL=""; TRAINED=0; DRY=0
for a in "$@"; do
  case $a in --trained) TRAINED=1 ;; --dry-run) DRY=1 ;; *) MODEL=$a ;; esac
done
case $MODE in sync|rtc) ;; *) echo "usage: switch.sh sync|rtc [hf_ckpt] [--trained] [--dry-run]"; exit 1 ;; esac
LOGF=$(ls -t "$R"/logs/lingbot_infer_*.log | head -1)
[ -n "$MODEL" ] || MODEL=$(grep "Policy loaded" "$LOGF" | tail -1 | sed 's/.*model_path=//')
[ -f "$MODEL/config.json" ] || { echo "no checkpoint at '$MODEL'"; exit 1; }
last=$(tail -1 "$R/logs/takeover_events.jsonl" | python3 -c 'import sys, json; print(json.loads(sys.stdin.read())["event"])')
[ "$last" = stop ] || { echo "a session is running (last takeover event: $last); stop it first"; exit 1; }
if [ "$TRAINED" = 1 ]; then
  grep -qxF "$MODEL" "$HOME/rtc/allow.txt" 2>/dev/null || echo "$MODEL" >> "$HOME/rtc/allow.txt"
fi
marked=no
{ [ -f "$MODEL/rtc_trained.json" ] || grep -qF "$MODEL" "$HOME/rtc/allow.txt" 2>/dev/null; } && marked=yes
echo "mode=$MODE model=$MODEL rtc_trained=$marked"
[ "$MODE" = rtc ] && [ "$marked" = no ] && echo "WARNING: not marked RTC-trained -> async without prefix (use --trained only for p10 checkpoints)"
[ "$DRY" = 1 ] && { echo "dry run: nothing changed"; exit 0; }

cp "$R/config/inference.$MODE.yaml" "$R/config/inference.yaml"
P=$(systemctl show lingbot-infer -p MainPID --value)
kill -KILL "$P"
for i in $(seq 1 40); do
  sleep 2
  N=$(systemctl show lingbot-infer -p MainPID --value); S=$(systemctl show lingbot-infer -p ActiveState --value)
  [ "$N" != "$P" ] && [ "$N" != 0 ] && [ "$S" = active ] && break
done
sleep 3
bash "$HOME/yolocue/lbcmd.sh" load_policy "{\"overrides\":{\"policy.model_path\":\"$MODEL\"},\"force\":true}" | grep -o '"accepted":[a-z]*' || true
NEWLOG=$(ls -t "$R"/logs/lingbot_infer_*.log | head -1)
for i in $(seq 1 60); do grep -q "Policy loaded" "$NEWLOG" && break; sleep 3; done
grep -E "Policy loaded|ERROR" "$NEWLOG" | tail -2 | cut -c1-220
echo "switched to $MODE (service pid $N). RTC lines appear in $NEWLOG at the next START: 'RTC on' / 'RTC chunk: ...'"
