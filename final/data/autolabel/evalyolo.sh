#!/bin/bash
# usage: evalyolo.sh RUN  (fetches ~/locany/yolo/runs/RUN/dets_first.json, scores on all and reserve-only episodes)
D=./la/eval
RC=~/lingbot-final-notes/experiments/real_detector/real_check; PY=~/.venvs/field/bin/python
for s in all reserve; do
  scp -q lbreal:locany/yolo/runs/$1/dets_first.json $D/$s/dets_$1.json
  (cd $D/$s && $PY $RC/stats.py dets_$1.json > /dev/null && echo "=== $s ===" && $PY $RC/evalgt.py dets_$1.json 0.25 | grep -v "^    ")
done
