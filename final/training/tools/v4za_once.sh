#!/bin/bash
# v4za_once.sh - ONE-TIME exception to the no-discard rule, by explicit user request (11/16 03:5x, via the data session):
#   "帮我改一下设置，我们训到15200，然后现在已经预热好了4条，帮我处理好以后重新从14500启动"
# As soon as the new batch is staged (any tag in tools/batches.txt whose sets are not in lists/v4z.txt): build lists/v4za.txt
# with mk_v4c (v4z's batches + the new one), configs/v4za.yaml = configs/v4z.yaml with that list and save_steps 100 (saves at
# v4 12600 ... 13200 = overall 15200; max_steps stays 30000), and restart from v4 global_step_12500 (overall 14500) right away -
# the steps trained on v4z after 12500 are given up on purpose. Fallback: configs/v4z_s100.yaml (v4z list, save_steps 100).
# On OK, auto_multi.sh is re-armed with FLOOR 13200, so no later batch restarts the run before the 15200 save.
# usage: setsid nohup bash tools/v4za_once.sh > logs/v4za_once.out 2>&1 < /dev/null &
W=$HOME/lingbot_train; R=/data/lingbot/runs/v4; D=/data/lingbot/data; B=$W/tools/batches.txt
CUR=v4z; NEW=v4za; FB=v4z_s100; ST=12500
LOG=$W/logs/v4za_once.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
sets_of() { grep -oE "^$D/[A-Za-z0-9_]+" "$D/STAGED_$1.txt" | sed "s#^$D/##" | sort -u; }
in_list() { case $1 in *t3_*) grep -qE "/${1}__[a-z]+\$" "$2" ;; *box*) grep -q "/${1}_named\$" "$2" ;; *) grep -q "/${1}\$" "$2" ;; esac; }
scan() { READY=""; while read -r t h; do case "$t" in ''|\#*) continue ;; esac
  [ -f "$D/STAGED_$t.txt" ] || continue; s=$(sets_of "$t"); [ -n "$s" ] || continue
  pend=0; for x in $s; do in_list "$x" "$W/lists/$CUR.txt" || pend=1; done; [ $pend = 1 ] && READY="$READY $t"; done < "$B"; }
cd "$W" || exit 1
for f in lists/$NEW.txt configs/$NEW.yaml configs/$FB.yaml logs/${NEW}_switch.log logs/mk_$NEW.out; do
  [ -e "$f" ] && { log "ABORT name taken ($f exists)"; exit 1; }
done
pgrep -f "bash tools/auto_multi.sh" > /dev/null && { log "ABORT auto_multi.sh is running (would restart at 13000); stop it first"; exit 1; }
[ -f "$R/checkpoints/global_step_$ST/model/.metadata" ] && [ -f "$R/checkpoints/global_step_$ST/optimizer/.metadata" ] || { log "ABORT no complete global_step_$ST"; exit 1; }
sed -e 's/^\(  *save_steps:\) *500$/\1 100/' configs/$CUR.yaml > configs/$FB.yaml
grep -qE '^ +save_steps: 100$' configs/$FB.yaml && grep -qE '^ +max_steps: 30000$' configs/$FB.yaml || { log "ABORT configs/$FB.yaml"; exit 1; }
log "armed: waiting for a staged batch not in lists/$CUR.txt; then restart from v4 global_step_$ST with save_steps 100"
while :; do scan; [ -n "$READY" ] && ! pgrep -f "bash tools/v4b_switch.sh" > /dev/null && break; sleep 10; done
log "ready:$READY"
ALL=""
for t in $READY; do
  cut -c1-200 "$D/STAGED_$t.txt" | tee -a "$LOG"
  for x in $(sets_of "$t"); do
    [ -f "$D/$x/meta/info.json" ] || { log "ABORT $D/$x has no meta/info.json"; exit 1; }
    case $x in
      *t3_*) ls -d "$D/${x}__"* > /dev/null 2>&1 || python tools/split_t3_subtasks.py "$D/$x" 2>&1 | tee -a "$LOG"
             ls -d "$D/${x}__"* > /dev/null 2>&1 || { log "ABORT split produced no subtask dirs for $x"; exit 1; } ;;
      *box*) if [ ! -f "$D/${x}_named/meta/info.json" ] || [ ! -f "$D/${x}_gen/meta/info.json" ]; then
               python tools/build_cue_prompts.py --bridge "$W/tools/yolocue_bridge.py" "$D/$x" >> "$LOG" 2>&1 || { log "ABORT build_cue_prompts failed for $x"; exit 1; }
             fi ;;
    esac
    ALL="$ALL $x"
  done
done
python tools/mk_v4c.py --write --out "$W/lists/$NEW.txt" > logs/mk_$NEW.out 2>&1 || { log "ABORT mk_v4c failed"; exit 1; }
grep -E "^group boost|^WARNING|^scale|recency|DAgger:" logs/mk_$NEW.out | cut -c1-220 | tee -a "$LOG"
grep -q "recency pairwise: OK" logs/mk_$NEW.out && grep -q "recency: newest.*OK" logs/mk_$NEW.out || { log "ABORT recency check failed"; exit 1; }
for x in $ALL; do in_list "$x" "lists/$NEW.txt" || { log "ABORT $x missing from lists/$NEW.txt"; exit 1; }; done
sed "s#lists/$CUR.txt#lists/$NEW.txt#" configs/$FB.yaml > configs/$NEW.yaml
grep -q "lists/$NEW.txt" configs/$NEW.yaml && grep -qE '^ +save_steps: 100$' configs/$NEW.yaml || { log "ABORT configs/$NEW.yaml"; exit 1; }
export FORGE_EXPORT_NAMES=$(cat lists/$NEW.txt lists/$CUR.txt | awk 'NF >= 2 {print $1}' | awk '!s[$0]++' | paste -sd,)
log "$NEW: $(wc -l < lists/$NEW.txt) lines; restarting from v4 global_step_$ST NOW (user's one-time exception); FORGE_EXPORT_NAMES=$FORGE_EXPORT_NAMES"
STEP=$ST WAIT=1 CFG=configs/$NEW.yaml FALLBACK=configs/$FB.yaml LOG=$W/logs/${NEW}_switch.log bash tools/v4b_switch.sh > logs/${NEW}_switch.out 2>&1
res=$(tail -1 "$W/logs/${NEW}_switch.log" | cut -c1-200); log "switch exited: $res"
case "$res" in *" OK: step"*) ;; *) log "ABORT switch did not report OK (fallback $FB from $ST?); auto_multi NOT re-armed"; exit 1 ;; esac
grep -q "CFG=$W/configs/$NEW.yaml" <(tail -1 "$R/launch.txt") || { log "ABORT last launch is not $NEW"; exit 1; }
setsid nohup bash tools/auto_multi.sh 13200 $NEW v4zb v4zc v4zd v4ze > logs/auto_multi.out 2>&1 < /dev/null &
sleep 3; log "re-armed auto_multi: $(pgrep -fa 'bash tools/auto_multi.sh' | cut -c1-80)"
