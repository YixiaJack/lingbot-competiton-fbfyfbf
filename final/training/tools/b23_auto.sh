#!/bin/bash
# 11/15 22:15: batch 23 (snapshot e94f77bf, eps 511-528, all task 2 normal teleop; processed by the data session). Same
# flow as b22_auto.sh. When STAGED_e94f lands: build cue prompts for the staged box set (own_t12_box_v13_newe94f, YOLO
# v13 cues), write lists/v4u.txt with mk_v4c (ladder v2: demos 23; caps 8% / 13%; group boost +4 pts with e94f in the
# group), verify every staged set is in the list, then restart. Restart point: the newest saved 500-step checkpoint if
# train.log is less than 300 steps past it (never earlier than v4 10500 = overall 12500, so the v4t steps are kept),
# otherwise the next save.
# usage: setsid nohup bash tools/b23_auto.sh > logs/v4u_auto.out 2>&1 < /dev/null &
W=$HOME/lingbot_train; R=/data/lingbot/runs/v4; LOG=$W/logs/v4u_auto.log; ST_F=/data/lingbot/data/STAGED_e94f.txt
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
python tools/mk_v4c.py --write --out "$W/lists/v4u.txt" > logs/mk_v4u.out 2>&1 || { log "ABORT mk_v4c failed"; exit 1; }
grep -E "^task-|^scale|recency|e94f|DAgger:" logs/mk_v4u.out | cut -c1-220 | tee -a "$LOG"
grep -q "recency pairwise: OK" logs/mk_v4u.out && grep -q "recency: newest.*OK" logs/mk_v4u.out || { log "ABORT recency check failed"; exit 1; }
for s in $SETS; do
  case $s in
    *t3_*) grep -qE "/${s}__[a-z]+\$" lists/v4u.txt || { log "ABORT $s missing from lists/v4u.txt (mk_v4c needs the name)"; exit 1; } ;;
    *box*) for v in named gen; do grep -q "/${s}_$v\$" lists/v4u.txt || { log "ABORT ${s}_$v missing from lists/v4u.txt"; exit 1; }; done ;;
    *) grep -q "/${s}\$" lists/v4u.txt || { log "ABORT $s missing from lists/v4u.txt (mk_v4c needs the name)"; exit 1; } ;;
  esac
done
sed "s#lists/v4t.txt#lists/v4u.txt#" configs/v4t.yaml > configs/v4u.yaml
grep -q "lists/v4u.txt" configs/v4u.yaml || { log "ABORT configs/v4u.yaml"; exit 1; }
s=$(step); base=$(( s / 500 * 500 ))
if [ $(( s - base )) -lt 300 ]; then ST=$base; else ST=$(( base + 500 )); fi
[ "$ST" -lt 10500 ] && ST=10500
log "v4u: $(wc -l < lists/v4u.txt) lines, $(grep -c e94f lists/v4u.txt) e94f; train.log at step $s -> switching from v4 global_step_$ST"
STEP=$ST WAIT=1 CFG=configs/v4u.yaml FALLBACK=configs/v4t.yaml LOG=$W/logs/v4u_switch.log bash tools/v4b_switch.sh > logs/v4u_switch.out 2>&1
log "switch exited: $(tail -1 "$W/logs/v4u_switch.log" | cut -c1-200)"
