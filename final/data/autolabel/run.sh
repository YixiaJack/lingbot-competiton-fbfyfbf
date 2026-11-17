#!/bin/bash
# Keeps label.py running and out of the way of the official inference service.
# usage: run.sh NAME [label.py args...]     pause: touch ~/locany/label/PAUSE    resume: rm it
cd ~/locany && source env/bin/activate
NAME=$1; shift
mkdir -p label/$NAME; LOG=label/$NAME/run.log
fails=0; waiting=0
while true; do
  if [ -f label/PAUSE ]; then sleep 15; continue; fi
  if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -gt 0 ]; then
    [ $waiting -eq 0 ] && echo "$(date '+%F %T') GPU in use by another process, waiting" >> $LOG
    waiting=1; sleep 60; continue
  fi
  waiting=0
  before=$(wc -l < label/$NAME/labels.jsonl 2>/dev/null || echo 0)
  python scripts/label.py --name $NAME "$@" 2>&1 | grep --line-buffered -v "Warning\|warn(\|GenerationMixin\|inherit\|auto-classes\|owner\|slow processor\|torch_dtype\|box_avg\|Loading checkpoint" >> $LOG
  rc=${PIPESTATUS[0]}
  [ $rc -eq 0 ] && { echo "$(date '+%F %T') finished" >> $LOG; break; }
  [ $rc -eq 3 ] && { sleep 30; continue; }
  after=$(wc -l < label/$NAME/labels.jsonl 2>/dev/null || echo 0)
  [ $((after - before)) -gt 500 ] && fails=0   # only count crashes that made no real progress
  fails=$((fails+1)); echo "$(date '+%F %T') crash rc=$rc ($fails/5)" >> $LOG
  [ $fails -ge 5 ] && { echo "$(date '+%F %T') giving up" >> $LOG; touch label/$NAME/FAILED; break; }
  sleep 30
done
