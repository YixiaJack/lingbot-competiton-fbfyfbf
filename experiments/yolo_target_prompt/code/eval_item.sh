#!/usr/bin/env bash
set -euo pipefail
ROOT=$SCRATCH/lingbot/final_yolo_20261111
HERE="$ROOT/code"
RUN="$1"; MODE="$2"; SOURCE="$3"; ITEM="$4"
BASE=$SCRATCH/lingbot
export LINGBOT_DIR="$BASE/dev/lingbot-vla-v2-dino"
export PYTHONPATH="$ROOT/deps:${PYTHONPATH:-}"
if [[ "$RUN" == base ]]; then
  CKPT="$BASE/runs/strong_scratch22k_cont8k/checkpoints/global_step_8000/hf_ckpt"
else
  CKPT="$ROOT/runs/$RUN/checkpoints/global_step_2000/hf_ckpt"
fi
OUT="$ROOT/results/${RUN}_${MODE}_${SOURCE}/$(printf '%04d' "$ITEM")"
if [[ -f "$OUT/result.json" ]]; then
  python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r.get("metric_version")==2 and r.get("scene_fingerprint")' "$OUT/result.json" && exit 0
fi
mkdir -p "$OUT"
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])')
setsid bash "$HERE/runtime.sh" model python "$HERE/server.py" --model_path "$CKPT" --use_length 50 \
  --use_bf16 True --use_fp32 False --use_compile True --port "$PORT" > "$OUT/server.log" 2>&1 &
SERVER=$!
cleanup() { kill -- "-$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; }
trap cleanup EXIT
READY=0
for ((n=0;n<240;n++)); do
  kill -0 "$SERVER" 2>/dev/null || { tail -30 "$OUT/server.log"; exit 1; }
  if (echo > "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then READY=1; break; fi
  sleep 5
done
[[ "$READY" == 1 ]] || { echo 'Server did not become ready in 20 minutes'; exit 4; }
timeout 3600 bash "$HERE/runtime.sh" sim python "$HERE/evaluate.py" --root "$ROOT" \
  --run "$RUN" --mode "$MODE" --source "$SOURCE" --item "$ITEM" --port "$PORT" > "$OUT/client.log" 2>&1
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r.get("metric_version")==2 and r.get("scene_fingerprint"); assert r["precision"]=="BF16"' "$OUT/result.json"
