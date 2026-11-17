#!/usr/bin/env bash
# One server per batch: RUN MODE SOURCE CHUNK -> eval items CHUNK*10 .. CHUNK*10+9 (compile cost paid once per batch).
set -euo pipefail
R=$SCRATCH/lingbot/final_real_20261112; BASE=$SCRATCH/lingbot; HERE=$R/code
RUN="$1"; MODE="$2"; SOURCE="$3"; CHUNK="$4"; MEMCAP="${5:-}"
TAG=""; [[ -n "$MEMCAP" ]] && { TAG="_mem$MEMCAP"; export LINGBOT_MEM_CAP_GB=$MEMCAP; }
export LINGBOT_DIR=$BASE/dev/lingbot-vla-v2-dino
export PYTHONPATH="$BASE/final_yolo_20261111/deps:${PYTHONPATH:-}"
STEPS=$(python3 -c "import json; print(json.load(open('$R/ab_runs.json'))['$RUN'][2])")
CKPT=$R/runs/$RUN/checkpoints/global_step_$STEPS/hf_ckpt
[[ -f $R/runs/$RUN/DONE ]] || { echo "run $RUN not done"; exit 2; }
items=$(seq $((CHUNK*10)) $((CHUNK*10+9)))
todo=""; for i in $items; do
  [[ -f $R/eval_seeds/$(printf %04d $i).json ]] || { echo "seed of item $i missing"; exit 2; }
  [[ -f $R/results/${RUN}_${MODE}_${SOURCE}${TAG}/$(printf %04d $i)/result.json ]] || todo="$todo $i"
done
[[ -z "$todo" ]] && exit 0
LOGD=$R/results/${RUN}_${MODE}_${SOURCE}${TAG}/logs; mkdir -p $LOGD
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])')
setsid bash $HERE/runtime.sh model python $HERE/server.py --model_path $CKPT --use_length 50 \
  --use_bf16 True --use_fp32 False --use_compile True --port $PORT > $LOGD/server_chunk$CHUNK.log 2>&1 &
SERVER=$!
cleanup() { kill -- "-$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; }
trap cleanup EXIT
for ((n=0;n<240;n++)); do
  kill -0 "$SERVER" 2>/dev/null || { tail -30 $LOGD/server_chunk$CHUNK.log; exit 1; }
  (echo > "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && break
  sleep 5
done
fail=0
for i in $todo; do
  timeout 2400 bash $HERE/runtime.sh sim python $HERE/eval_real.py --root $R --run $RUN --mode $MODE --source $SOURCE \
    --item $i --port $PORT --tag "$TAG" >> $LOGD/client_chunk$CHUNK.log 2>&1 || fail=1
done
exit $fail
