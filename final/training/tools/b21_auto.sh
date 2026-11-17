#!/bin/bash
# 11/15 16:50: batch 21 (snapshot 77235417, eps 475-485, all task 3; processed by the data session). When STAGED_7723 lands:
# split every staged task-3 set into __{grasp,sweep,dump,home} (split_t3_subtasks.py, by the source_map subtask = the
# episode instruction prefix), build cue prompts for staged box sets, write lists/v4q.txt with mk_v4c (ladder v2: demos 21,
# takeover DAgger 31.5, task-3 DAgger cap 13% of task 3), verify every staged set is in the list, then restart. Restart
# point as in b20_auto: the newest saved 500-step checkpoint if train.log is less than 300 steps past it (never earlier
# than v4 8500 = overall 10500), otherwise the next save.
# usage: setsid nohup bash tools/b21_auto.sh > logs/v4q_auto.out 2>&1 < /dev/null &
W=$HOME/lingbot_train; R=/data/lingbot/runs/v4; LOG=$W/logs/v4q_auto.log; ST_F=/data/lingbot/data/STAGED_7723.txt
D=/data/lingbot/data
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
step() { tail -c 6000 "$R/train.log" | tr '\r' '\n' | grep -ao 'Step [0-9]*,' | tail -1 | tr -dc 0-9; }
log "armed: waiting for $ST_F and for other switches to finish"
until [ -f "$ST_F" ]; do sleep 20; done
sleep 10
while pgrep -f "bash tools/v4b_switch.sh" > /dev/null; do sleep 15; done
log "STAGED present, no other switch running"
cut -c1-200 "$ST_F" | tee -a "$LOG"
cd "$W" || exit 1
SETS=$(grep -oE "^$D/[A-Za-z0-9_]+" "$ST_F" | sed "s#^$D/##" | sort -u)
[ -n "$SETS" ] || { log "ABORT no set paths in $ST_F"; exit 1; }
for s in $SETS; do
  [ -f "$D/$s/meta/info.json" ] || { log "ABORT $D/$s has no meta/info.json"; exit 1; }
  case $s in
    *t3_*)
      if ! ls -d "$D/${s}__"* > /dev/null 2>&1; then
        python tools/split_t3_subtasks.py "$D/$s" 2>&1 | tee -a "$LOG"
      fi
      ls -d "$D/${s}__"* > /dev/null 2>&1 || { log "ABORT split produced no subtask dirs for $s"; exit 1; } ;;
    *box*)
      if [ ! -f "$D/${s}_named/meta/info.json" ] || [ ! -f "$D/${s}_gen/meta/info.json" ]; then
        python tools/build_cue_prompts.py --bridge "$W/tools/yolocue_bridge.py" "$D/$s" >> "$LOG" 2>&1 \
          || { log "ABORT build_cue_prompts failed for $s"; exit 1; }
      fi ;;
  esac
done
python tools/mk_v4c.py --write --out "$W/lists/v4q.txt" > logs/mk_v4q.out 2>&1 || { log "ABORT mk_v4c failed"; exit 1; }
grep -E "^task-|^scale|recency|7723|DAgger:" logs/mk_v4q.out | cut -c1-220 | tee -a "$LOG"
grep -q "recency pairwise: OK" logs/mk_v4q.out && grep -q "recency: newest.*OK" logs/mk_v4q.out || { log "ABORT recency check failed"; exit 1; }
for s in $SETS; do
  case $s in
    *t3_*) grep -qE "/${s}__[a-z]+\$" lists/v4q.txt || { log "ABORT $s missing from lists/v4q.txt (mk_v4c needs the name)"; exit 1; } ;;
    *box*) for v in named gen; do grep -q "/${s}_$v\$" lists/v4q.txt || { log "ABORT ${s}_$v missing from lists/v4q.txt"; exit 1; }; done ;;
    *) grep -q "/${s}\$" lists/v4q.txt || { log "ABORT $s missing from lists/v4q.txt (mk_v4c needs the name)"; exit 1; } ;;
  esac
done
sed "s#lists/v4p.txt#lists/v4q.txt#" configs/v4p.yaml > configs/v4q.yaml
grep -q "lists/v4q.txt" configs/v4q.yaml || { log "ABORT configs/v4q.yaml"; exit 1; }
s=$(step); base=$(( s / 500 * 500 ))
if [ $(( s - base )) -lt 300 ]; then ST=$base; else ST=$(( base + 500 )); fi
[ "$ST" -lt 8500 ] && ST=8500
log "v4q: $(wc -l < lists/v4q.txt) lines, $(grep -c 7723 lists/v4q.txt) 7723; train.log at step $s -> switching from v4 global_step_$ST"
STEP=$ST WAIT=1 CFG=configs/v4q.yaml FALLBACK=configs/v4p.yaml LOG=$W/logs/v4q_switch.log bash tools/v4b_switch.sh > logs/v4q_switch.out 2>&1
log "switch exited: $(tail -1 "$W/logs/v4q_switch.log" | cut -c1-200)"
