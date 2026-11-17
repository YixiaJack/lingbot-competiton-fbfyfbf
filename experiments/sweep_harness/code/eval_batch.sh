#!/usr/bin/env bash
# One LingBot server per batch: RUN HARNESS CHUNK [STEP] -> eval items CHUNK*5 .. CHUNK*5+4 with that harness.
# Also RUN HARNESS_CHUNK [STEP] (one pool item, e.g. v2_3): the pool's phases file is split on '|', so the phase command
# cannot split the item with a shell pipe.
# All harnesses use the same checkpoint, scenes (eval_seeds/) and VLA noise seeds.
set -euo pipefail
R=${R:-$SCRATCH/lingbot/sweep_harness_20261114}; BASE=$SCRATCH/lingbot; HERE=$SCRATCH/lingbot/sweep_harness_20261114/code
if [[ "$2" =~ ^([a-z0-9]+)_([0-9]+)$ ]]; then
  RUN="$1"; HARNESS="${BASH_REMATCH[1]}"; CHUNK="${BASH_REMATCH[2]}"; STEP="${3:-5000}"
else
  RUN="$1"; HARNESS="$2"; CHUNK="$3"; STEP="${4:-5000}"
fi
TAG="${TAG:-}"
export LINGBOT_DIR=$BASE/dev/lingbot-vla-v2-dino
CKPT=$R/runs/$RUN/checkpoints/global_step_$STEP/hf_ckpt
[[ -f $CKPT/model.safetensors.index.json ]] || { echo "checkpoint $CKPT missing"; exit 2; }
todo=""; for i in $(seq $((CHUNK*5)) $((CHUNK*5+4))); do
  [[ -f $R/eval_seeds/$(printf %04d $i).json ]] || { echo "seed of item $i missing"; exit 2; }
  [[ -f $R/results/${RUN}_${HARNESS}${TAG}/$(printf %04d $i)/result.json ]] || todo="$todo $i"
done
[[ -z "$todo" ]] && exit 0
LOGD=$R/results/${RUN}_${HARNESS}${TAG}/logs; mkdir -p $LOGD
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])')
setsid bash $HERE/runtime.sh model python $BASE/final_real_20261112/code/server.py --model_path $CKPT --use_length 50 \
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
  timeout 3000 bash $HERE/runtime.sh sim python $HERE/rollout.py --root $R --run $RUN --harness $HARNESS --item $i \
    --port $PORT --tag "$TAG" ${LIMIT:+--limit $LIMIT} >> $LOGD/client_chunk$CHUNK.log 2>&1 || fail=1   # LIMIT: tests only
done
exit $fail
