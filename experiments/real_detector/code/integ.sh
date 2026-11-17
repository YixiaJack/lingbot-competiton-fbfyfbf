#!/usr/bin/env bash
# Plug-in check on the official 6B base: server (BF16, stock LingBot deploy) + competition-like scenes + cues.
set -euo pipefail
R=$SCRATCH/lingbot/final_real_20261112; BASE=$SCRATCH/lingbot; HERE=$R/code
SOURCE="${1:-oracle}"
export LINGBOT_DIR=$BASE/dev/lingbot-vla-v2-dino
export PYTHONPATH="$BASE/final_yolo_20261111/deps:${PYTHONPATH:-}"
RUN=$R/integ/base_run; mkdir -p $RUN/checkpoints/global_step_0
[[ -e $RUN/checkpoints/global_step_0/hf_ckpt ]] || ln -s $BASE/ckpts/lingbot-vla-v2-6b $RUN/checkpoints/global_step_0/hf_ckpt
# the deploy server reads <run>/lingbotvla_cli.yaml: reuse the prelim training config, point the model at the base
bash $HERE/runtime.sh model python -c "
import yaml; c=yaml.safe_load(open('$BASE/runs/strong_scratch22k_cont8k/lingbotvla_cli.yaml'))
c['model']['model_path']='$BASE/ckpts/lingbot-vla-v2-6b'
yaml.safe_dump(c, open('$RUN/lingbotvla_cli.yaml','w'), sort_keys=False)"
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])')
mkdir -p $R/integ/logs
setsid bash $HERE/runtime.sh model python $HERE/server.py --model_path $RUN/checkpoints/global_step_0/hf_ckpt \
  --use_length 50 --use_bf16 True --use_fp32 False --use_compile True --port $PORT > $R/integ/logs/server_$SOURCE.log 2>&1 &
SERVER=$!
cleanup() { kill -- "-$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; }
trap cleanup EXIT
for ((n=0;n<240;n++)); do
  kill -0 "$SERVER" 2>/dev/null || { tail -40 $R/integ/logs/server_$SOURCE.log; exit 1; }
  (echo > "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && break
  sleep 5
done
for MODE in box coords; do
  timeout 3600 bash $HERE/runtime.sh sim python $HERE/integ_6b.py --root $R --port $PORT --source $SOURCE --mode $MODE --episodes 4
done
