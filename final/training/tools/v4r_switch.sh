#!/bin/bash
# v4r_switch.sh (dev machine): once v4 checkpoint global_step_$STEP (default 1000 = overall 3000) is fully saved,
# restart v4 from it with code_v4r (code_v4s + p10 training-time RTC).
# - Full resume: model, optimizer, LR schedule, dataloader and step counter. Same config, list and LB_*/FORGE_* env as
#   the last v4 launch (read from launch.txt), plus LB_RTC_MAX_DELAY / LB_RTC_P0.
# - Leaves step_offset, ckpt_guard and the board alone. max_steps stays 30000.
# - If the new run dies before step STEP+2, relaunch the same resume with code_v4s and no RTC, so the GPUs don't sit idle.
# usage: setsid nohup bash v4r_switch.sh > ~/lingbot_train/logs/v4r_switch.out 2>&1 &
set -u
W=$HOME/lingbot_train; D=/data/lingbot/runs/v4; STEP=${STEP:-1000}; C=$D/checkpoints/global_step_$STEP
LOG=$W/logs/v4r_switch.log
RTC_MAX=${RTC_MAX:-24}; RTC_P0=${RTC_P0:-0.3}
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
running() { pgrep -f "[t]asks/vla/train_lingbotvla.py $W/configs/v4.yaml" >/dev/null; }

log "waiting for $C (DCP model/optimizer/extra_state + hf_ckpt flattened) and step $((STEP + 1)) in train.log"
until [ -f "$C/model/.metadata" ] && [ -f "$C/optimizer/.metadata" ] && [ -d "$C/extra_state" ] \
    && [ -f "$C/hf_ckpt/model.safetensors.index.json" ] \
    && grep -aq "flattened .* -> $C/hf_ckpt" "$D/train.log" \
    && grep -aqE "[^0-9]$((STEP + 1))/30000" "$D/train.log"; do
  sleep 20
done
log "checkpoint complete: $(du -sh "$C" | cut -f1), hf_ckpt $(du -sh "$C/hf_ckpt" | cut -f1)"

ENVLINE=$(tail -1 "$D/launch.txt" | sed -n 's/.* ENV=//p')
[ -n "$ENVLINE" ] || { log "ABORT: no ENV= in the last launch.txt line; v4 left running"; exit 1; }
PID=$(cat "$D/pid"); PG=$(ps -o pgid= -p "$PID" | tr -d ' ')
log "stopping v4: torchrun pid $PID pgid $PG"
kill -TERM -- "-$PG"
for i in $(seq 1 90); do running || break; sleep 2; done
if running; then log "still running after 180 s: SIGKILL"; pkill -KILL -f "[t]asks/vla/train_lingbotvla.py $W/configs/v4.yaml"; sleep 10; fi
for i in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)
  [ "$busy" -eq 0 ] && break; sleep 2
done
log "GPUs free (busy=$busy)"

launch() {   # $1 = code dir, $2 = extra env ("" for none)
  ( cd "$W" && unset LB_RESUME_MODEL_ONLY && export $ENVLINE $2 && \
    CODE=$1 RUNS=/data/lingbot/runs bash "$W/tools/train_run.sh" v4 "$W/configs/v4.yaml" 0,1,2,3,4,5,6,7 \
      --train.load_checkpoint_path "$C" ) 2>&1 | tee -a "$LOG"
}
launch "$W/code_v4r" "LB_RTC_MAX_DELAY=$RTC_MAX LB_RTC_P0=$RTC_P0"
log "launched code_v4r with LB_RTC_MAX_DELAY=$RTC_MAX LB_RTC_P0=$RTC_P0, ENV: $ENVLINE"

check() {   # wait for a successful load and step STEP+2 in the new train.log; 0 = ok
  for i in $(seq 1 120); do
    sleep 15
    if grep -aqE "[^0-9]$((STEP + 2))/30000" "$D/train.log"; then return 0; fi
    running || { sleep 5; running || return 1; }
  done
  return 1
}
if check; then
  log "OK: $(grep -ac '\[p10\] training-time RTC on' "$D/train.log") ranks report p10 on; $(grep -a 'Load distributed checkpoint' "$D/train.log" | tail -1 | cut -c1-200)"
  exit 0
fi
log "FAILED: code_v4r run did not reach step $((STEP + 2)). Tail of train.log:"
tail -c 3000 "$D/train.log" | tr '\r' '\n' | tail -30 >> "$LOG"
running && { pkill -TERM -f "[t]asks/vla/train_lingbotvla.py $W/configs/v4.yaml"; sleep 60; running && pkill -KILL -f "[t]asks/vla/train_lingbotvla.py $W/configs/v4.yaml"; sleep 10; }
cp "$D/train.log" "$W/logs/v4r_failed_train.log" 2>/dev/null
launch "$W/code_v4s" ""
log "FALLBACK: relaunched code_v4s without RTC from $C"
check && log "FALLBACK OK" || log "FALLBACK ALSO FAILED: needs a person"
