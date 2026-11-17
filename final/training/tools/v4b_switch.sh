#!/bin/bash
# v4b_switch.sh (dev): 11/15, user: from overall step 4500 (v4 global_step_2500) train on lists/v4b.txt = the boxfix /
# v7a box sets with box-referencing prompts (*_named, *_gen; experiments/autolabel/build_cue_prompts.py) + today's own
# data (own_t12_new1115, own_t3_sub_new1115). Stops the running v4 and relaunches the same run dir with code_v4r and the
# last launch's env (RTC); model + optimizer resumed, dataloader state NOT resumed (the list changed). If the new run
# does not reach step STEP+2, it relaunches configs/v4n.yaml from the same checkpoint so the GPUs don't sit idle.
# usage: setsid nohup bash v4b_switch.sh > ~/lingbot_train/logs/v4b_switch.out 2>&1 < /dev/null &
set -u
W=$HOME/lingbot_train; R=/data/lingbot/runs/v4; STEP=${STEP:-2500}; C=$R/checkpoints/global_step_$STEP
CFG=${CFG:-configs/v4b.yaml}; FALLBACK=${FALLBACK:-configs/v4n.yaml}; LOG=${LOG:-$W/logs/v4b_switch.log}
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
running() { pgrep -f "[t]asks/vla/train_lingbotvla.py" >/dev/null; }
step() { tail -c 6000 "$R/train.log" | tr '\r' '\n' | grep -ao 'Step [0-9]*,' | tail -1 | tr -dc 0-9; }

complete() { [ -f "$C/model/.metadata" ] && [ -f "$C/optimizer/.metadata" ] && [ -d "$C/extra_state" ] \
  && grep -aq "flattened .* -> $C/hf_ckpt" "$R/train.log"*; }
if [ "${WAIT:-0}" = 1 ]; then   # arm ahead of the save: wait for the complete checkpoint and step STEP+2 in train.log
  log "waiting for $C and step $((STEP + 2))"
  until complete && s=$(step) && [ -n "$s" ] && [ "$s" -ge $((STEP + 2)) ]; do sleep 20; done
fi
complete || { log "ABORT: $C not complete"; exit 1; }
LIST=$(grep -E '^\s*train_path:' "$W/$CFG" | awk '{print $2}')
[ -f "$LIST" ] || { log "ABORT: list $LIST of $CFG missing"; exit 1; }
while read -r robot path; do
  [ -z "$path" ] && continue
  [ -f "$path/meta/info.json" ] || { log "ABORT: $path (in $LIST) has no meta/info.json"; exit 1; }
done < "$LIST"
ENVS=$(tail -1 "$R/launch.txt" | sed 's/.* ENV=//')
case "$ENVS" in *LB_RTC_MAX_DELAY=24*LB_RTC_P0=0.3*) ;; *) log "ABORT: last launch ENV lacks RTC: $ENVS"; exit 1 ;; esac
log "checkpoint $C complete; list $LIST ($(wc -l < "$LIST") lines) all present; train.log at step $(step)"

PG=$(cat "$R/pid"); log "stopping process group $PG"; kill -TERM -- "-$PG"
for i in $(seq 1 90); do running || break; sleep 2; done
running && { log "still alive after 180 s: SIGKILL"; pkill -KILL -f "[t]asks/vla/train_lingbotvla.py"; sleep 10; }
log "gpu after stop: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"

launch() {   # $1 = config
  ( cd "$W" && unset LB_RESUME_MODEL_ONLY && env $ENVS CODE=$W/code_v4r RUNS=/data/lingbot/runs bash tools/train_run.sh v4 "$1" \
      0,1,2,3,4,5,6,7 --train.load_checkpoint_path "$C" --train.resume_dataloader_state false ) 2>&1 | tee -a "$LOG"
}
ok() {   # the new run loaded the checkpoint and reached STEP+2
  for i in $(seq 1 120); do
    sleep 15
    s=$(step); [ -n "$s" ] && [ "$s" -ge $((STEP + 2)) ] && return 0
    running || { sleep 5; running || return 1; }
  done
  return 1
}
launch "$CFG"; log "launched $CFG: $(tail -1 "$R/launch.txt" | cut -c1-250)"
if ok; then
  log "OK: step $(step); $(grep -ac '\[p10\] training-time RTC on' "$R/train.log") ranks report p10; $(grep -a 'Load distributed checkpoint' "$R/train.log" | tail -1 | cut -c1-160)"
  exit 0
fi
log "FAILED: $CFG did not reach step $((STEP + 2)); tail:"; tail -c 3000 "$R/train.log" | tr '\r' '\n' | tail -25 >> "$LOG"
running && { pkill -TERM -f "[t]asks/vla/train_lingbotvla.py"; sleep 60; running && pkill -KILL -f "[t]asks/vla/train_lingbotvla.py"; sleep 10; }
cp "$R/train.log" "$W/logs/v4b_failed_train.log" 2>/dev/null
launch "$FALLBACK"; log "FALLBACK: relaunched $FALLBACK from $C"
ok && log "FALLBACK OK at step $(step)" || log "FALLBACK ALSO FAILED: needs a person"
