#!/bin/bash
# 11/15 16:20, user (via data session): batch 20 (snapshot 908aa6e5, eps 439-474) should go in at the overall-10500 switch.
# When STAGED_908a lands: build cue prompts for every staged box set, write lists/v4p.txt with mk_v4c (ladder v2: demos 20,
# DAgger 30, DAgger total cap 8%), verify every staged set is in the list, then restart. Restart point: the newest saved
# 500-step checkpoint if train.log is less than 300 steps past it (never earlier than v4 8500 = overall 10500), otherwise
# the next save.
# usage: setsid nohup bash tools/b20_auto.sh > logs/v4p_auto.out 2>&1 < /dev/null &
W=$HOME/lingbot_train; R=/data/lingbot/runs/v4; LOG=$W/logs/v4p_auto.log; ST_F=/data/lingbot/data/STAGED_908a.txt
D=/data/lingbot/data
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
step() { tail -c 6000 "$R/train.log" | tr '\r' '\n' | grep -ao 'Step [0-9]*,' | tail -1 | tr -dc 0-9; }
log "armed: waiting for $ST_F and for other switches to finish"
until [ -f "$ST_F" ]; do sleep 20; done
sleep 10
while pgrep -f "bash tools/v4b_switch.sh" > /dev/null; do sleep 15; done
log "STAGED present, no other switch running"
cat "$ST_F" | cut -c1-200 | tee -a "$LOG"
cd "$W" || exit 1
SETS=$(grep -oE "^$D/[A-Za-z0-9_]+" "$ST_F" | sed "s#^$D/##" | sort -u)
[ -n "$SETS" ] || { log "ABORT no set paths in $ST_F"; exit 1; }
for s in $SETS; do
  case $s in *box*)
    if [ ! -f "$D/${s}_named/meta/info.json" ] || [ ! -f "$D/${s}_gen/meta/info.json" ]; then
      python tools/build_cue_prompts.py --bridge "$W/tools/yolocue_bridge.py" "$D/$s" >> "$LOG" 2>&1 \
        || { log "ABORT build_cue_prompts failed for $s"; exit 1; }
    fi ;;
  esac
done
python tools/mk_v4c.py --write --out "$W/lists/v4p.txt" > logs/mk_v4p.out 2>&1 || { log "ABORT mk_v4c failed"; exit 1; }
grep -E "^task-|^scale|recency|908a|DAgger:" logs/mk_v4p.out | cut -c1-220 | tee -a "$LOG"
grep -q "recency pairwise: OK" logs/mk_v4p.out && grep -q "recency: newest.*OK" logs/mk_v4p.out || { log "ABORT recency check failed"; exit 1; }
for s in $SETS; do
  case $s in
    *box*) for v in named gen; do grep -q "/${s}_$v\$" lists/v4p.txt || { log "ABORT ${s}_$v missing from lists/v4p.txt"; exit 1; }; done ;;
    *) grep -qE "/${s}(__[a-z]+)?\$" lists/v4p.txt || { log "ABORT $s missing from lists/v4p.txt (mk_v4c needs the name)"; exit 1; } ;;
  esac
done
sed "s#lists/v4o.txt#lists/v4p.txt#" configs/v4o.yaml > configs/v4p.yaml
grep -q "lists/v4p.txt" configs/v4p.yaml || { log "ABORT configs/v4p.yaml"; exit 1; }
s=$(step); base=$(( s / 500 * 500 ))
if [ $(( s - base )) -lt 300 ]; then ST=$base; else ST=$(( base + 500 )); fi
[ "$ST" -lt 8500 ] && ST=8500
log "v4p: $(wc -l < lists/v4p.txt) lines, $(grep -c 908a lists/v4p.txt) 908a; train.log at step $s -> switching from v4 global_step_$ST"
STEP=$ST WAIT=1 CFG=configs/v4p.yaml FALLBACK=configs/v4o.yaml LOG=$W/logs/v4p_switch.log bash tools/v4b_switch.sh > logs/v4p_switch.out 2>&1
log "switch exited: $(tail -1 "$W/logs/v4p_switch.log" | cut -c1-200)"
