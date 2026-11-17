#!/bin/bash
# auto_multi.sh FLOOR CUR NEW [NEW2 ...]      e.g. auto_multi.sh 10500 v4v v4w v4x v4y v4z
# 11/15 22:40 user rule (via the data session): when two or more newly collected batches are waiting untrained, put them
# ALL into ONE restart from the nearest checkpoint, not one restart per batch. Replaces the per-batch bNN_auto.sh scripts.
#
# Pending batches: the tags in tools/batches.txt (from batch 24 on; mk_v4c reads the same file for the ladder weight)
#   - ready = STAGED_<tag>.txt exists and none of its sets is in the current list (lists/CUR.txt) yet;
#   - a 2nd word "hold" is ignored since 11/15 23:3x: with the no-discard rule a restart happens right at a fresh save,
#     and waiting there for a batch that is not staged yet would throw away the steps trained meanwhile.
# Trigger: at least one ready batch. Then
#   1. restart point = the NEXT 500-step save after the current step (11/15 23:2x user rule: never discard training
#      progress; never below FLOOR); wait until that checkpoint is complete and the run is 2 steps past it;
#   2. re-read the pending batches right away (a batch staged during the wait joins this restart; one not staged yet
#      joins the next save) - never wait once the checkpoint is complete;
#   3. split staged task-3 sets into __{subtask}, build _named/_gen for staged box sets, write lists/NEW.txt with mk_v4c,
#      check recency + that every staged set is in the list, configs/NEW.yaml = configs/CUR.yaml with the new list;
#   4. one restart with v4b_switch.sh (model + optimizer resumed, dataloader not), fallback configs/CUR.yaml;
#   5. on success CUR=NEW, FLOOR=that step, take the next name, loop. Exits when the names run out or on an ABORT.
# usage: setsid nohup bash tools/auto_multi.sh 10500 v4v v4w v4x v4y v4z > logs/auto_multi.out 2>&1 < /dev/null &
W=$HOME/lingbot_train; R=/data/lingbot/runs/v4; D=/data/lingbot/data; B=$W/tools/batches.txt
FLOOR=${1:?floor step}; CUR=${2:?current list name}; shift 2; NAMES=("$@")
[ ${#NAMES[@]} -gt 0 ] || { echo "need at least one new list name"; exit 1; }
LOG=$W/logs/auto_multi.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
step() { tail -c 6000 "$R/train.log" | tr '\r' '\n' | grep -ao 'Step [0-9]*,' | tail -1 | tr -dc 0-9; }
sets_of() { grep -oE "^$D/[A-Za-z0-9_]+" "$D/STAGED_$1.txt" | sed "s#^$D/##" | sort -u; }
in_list() {   # set $1 (unsplit name) is in list file $2
  case $1 in
    *t3_*) grep -qE "/${1}__[a-z]+\$" "$2" ;;
    *box*) grep -q "/${1}_named\$" "$2" ;;
    *) grep -q "/${1}\$" "$2" ;;
  esac
}
scan() {   # sets READY (space-separated tags) and HELD (count)
  READY=""; HELD=0
  while read -r t h; do
    case "$t" in ''|\#*) continue ;; esac
    if [ -f "$D/STAGED_$t.txt" ]; then
      s=$(sets_of "$t"); [ -n "$s" ] || continue
      pend=0; for x in $s; do in_list "$x" "$W/lists/$CUR.txt" || pend=1; done
      [ $pend = 1 ] && READY="$READY $t"
    elif [ "$h" = hold ]; then
      HELD=$((HELD + 1))
    fi
  done < "$B"
}
cd "$W" || exit 1
for n in "${NAMES[@]}"; do
  for f in lists/$n.txt configs/$n.yaml logs/${n}_switch.log logs/mk_$n.out; do
    [ -e "$f" ] && { log "ABORT name $n taken ($f exists)"; exit 1; }
  done
done
log "armed: FLOOR $FLOOR, CUR $CUR, names ${NAMES[*]}"
for NEW in "${NAMES[@]}"; do
  log "[$NEW] waiting for staged batches from $B (current list $CUR)"
  while :; do
    scan
    [ -n "$READY" ] && ! pgrep -f "bash tools/v4b_switch.sh" > /dev/null && break
    sleep 20
  done
  log "[$NEW] ready:$READY"
  # 11/15 23:2x user: "每次加入新的预热数据的时候，一定不要舍弃当前的训练进度，而是统一加到下一次checkpoint保存之后再进行加入所有的预热数据训练"
  # -> never go back to an older save: always the NEXT save after the current step (the old "<300 steps past the newest
  # save -> restart from it" rule threw away 249 + 56 + 127 steps on 11/15 22:14 / 22:24 / 22:45).
  s=$(step); ST=$(( s / 500 * 500 + 500 ))
  [ "$ST" -lt "$FLOOR" ] && ST=$FLOOR
  C=$R/checkpoints/global_step_$ST
  log "[$NEW] train.log at step $s -> restart point v4 global_step_$ST; waiting for it to be complete"
  until [ -f "$C/model/.metadata" ] && [ -f "$C/optimizer/.metadata" ] && [ -d "$C/extra_state" ] \
      && grep -aq "flattened .* -> $C/hf_ckpt" "$R/train.log"* && s=$(step) && [ -n "$s" ] && [ "$s" -ge $((ST + 2)) ]; do
    sleep 20
  done
  scan   # no waiting here: every step trained after the save would be discarded (11/15 23:3x)
  log "[$NEW] checkpoint $ST complete; batches in this restart:$READY"
  ALL=""
  for t in $READY; do
    cut -c1-200 "$D/STAGED_$t.txt" | tee -a "$LOG"
    for x in $(sets_of "$t"); do
      [ -f "$D/$x/meta/info.json" ] || { log "ABORT $D/$x has no meta/info.json"; exit 1; }
      case $x in
        *t3_*) ls -d "$D/${x}__"* > /dev/null 2>&1 || python tools/split_t3_subtasks.py "$D/$x" 2>&1 | tee -a "$LOG"
               ls -d "$D/${x}__"* > /dev/null 2>&1 || { log "ABORT split produced no subtask dirs for $x"; exit 1; } ;;
        *box*) if [ ! -f "$D/${x}_named/meta/info.json" ] || [ ! -f "$D/${x}_gen/meta/info.json" ]; then
                 python tools/build_cue_prompts.py --bridge "$W/tools/yolocue_bridge.py" "$D/$x" >> "$LOG" 2>&1 \
                   || { log "ABORT build_cue_prompts failed for $x"; exit 1; }
               fi ;;
      esac
      ALL="$ALL $x"
    done
  done
  python tools/mk_v4c.py --write --out "$W/lists/$NEW.txt" > logs/mk_$NEW.out 2>&1 || { log "ABORT mk_v4c failed"; exit 1; }
  grep -E "^group boost|^WARNING|^scale|recency|DAgger:" logs/mk_$NEW.out | cut -c1-220 | tee -a "$LOG"
  grep -q "^group boost:" logs/mk_$NEW.out || log "[$NEW] WARN: no 'group boost:' line (boost skipped?) - continuing"
  grep -q "recency pairwise: OK" logs/mk_$NEW.out && grep -q "recency: newest.*OK" logs/mk_$NEW.out || { log "ABORT recency check failed"; exit 1; }
  for x in $ALL; do
    in_list "$x" "lists/$NEW.txt" || { log "ABORT $x missing from lists/$NEW.txt (tag not in tools/batches.txt?)"; exit 1; }
    case $x in *box*) grep -q "/${x}_gen\$" "lists/$NEW.txt" || { log "ABORT ${x}_gen missing from lists/$NEW.txt"; exit 1; } ;; esac
  done
  sed "s#lists/$CUR.txt#lists/$NEW.txt#" configs/$CUR.yaml > configs/$NEW.yaml
  grep -q "lists/$NEW.txt" configs/$NEW.yaml || { log "ABORT configs/$NEW.yaml"; exit 1; }
  log "[$NEW] $(wc -l < lists/$NEW.txt) lines; switching from v4 global_step_$ST"
  # 11/16 02:19 v4y died at launch: train_lingbotvla.py puts the first word of every list line into ONE env var
  # (FORGE_EXPORT_NAMES = "agilex_cobot_magic,agilex_cobot_magic,..." x lines); at 7715 lines that is > 128 KB (Linux
  # MAX_ARG_STRLEN), so inductor's subprocess exec failed with E2BIG. The code only uses the de-duplicated names and keeps a
  # preset value (setdefault), so preset the de-duplicated names of the new and the fallback list: same export, short env.
  export FORGE_EXPORT_NAMES=$(cat lists/$NEW.txt lists/$CUR.txt | awk 'NF >= 2 {print $1}' | awk '!s[$0]++' | paste -sd,)
  log "[$NEW] FORGE_EXPORT_NAMES=$FORGE_EXPORT_NAMES"
  STEP=$ST WAIT=1 CFG=configs/$NEW.yaml FALLBACK=configs/$CUR.yaml LOG=$W/logs/${NEW}_switch.log bash tools/v4b_switch.sh > logs/${NEW}_switch.out 2>&1
  res=$(tail -1 "$W/logs/${NEW}_switch.log" | cut -c1-200)
  log "[$NEW] switch exited: $res"
  case "$res" in *" OK: step"*) ;; *) log "ABORT [$NEW] switch did not report OK; stopping"; exit 1 ;; esac
  grep -q "CFG=$W/configs/$NEW.yaml" <(tail -1 "$R/launch.txt") || { log "ABORT [$NEW] last launch is not $NEW (fallback?)"; exit 1; }
  CUR=$NEW; FLOOR=$ST
done
log "names used up; exiting"
