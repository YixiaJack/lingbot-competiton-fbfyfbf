#!/bin/bash
# usage: train_run.sh RUN CONFIG GPUS [extra --key value ...]
#   RUN    name under ~/lingbot_train/runs (log: runs/RUN/train.log, pid: runs/RUN/pid)
#   CONFIG yaml made by tools/mkcfg.py
#   GPUS   e.g. 0,1,2,3,4,5,6,7
#   CODE   env var: code root (default /opt/lingbot-vla-v2; the optimized copy lives in ~/lingbot_train/code)
. ~/lingbot_train/tools/env.sh
RUN=$1; CFG=$(realpath "$2"); GPUS=$3; shift 3
CODE=${CODE:-$LB}
export CUDA_VISIBLE_DEVICES=$GPUS
N=$(echo "$GPUS" | tr ',' '\n' | wc -l)
D=$W/runs/$RUN; mkdir -p "$D"; cd "$D" || exit 1
if [ "$CODE" != "$LB" ]; then export PYTHONPATH=$CODE${PYTHONPATH:+:$PYTHONPATH}; fi
PORT=$((62600 + RANDOM % 300))
echo "$(date -Is) RUN=$RUN CFG=$CFG GPUS=$GPUS CODE=$CODE PORT=$PORT ARGS=$*" >> "$D/launch.txt"
setsid nohup torchrun --nproc-per-node "$N" --master-port "$PORT" "$CODE/tasks/vla/train_lingbotvla.py" "$CFG" \
  --train.output_dir "$D" "$@" > "$D/train.log" 2>&1 < /dev/null &
echo $! > "$D/pid"
echo "started $RUN pid $(cat "$D/pid") on $N GPUs, log $D/train.log"
