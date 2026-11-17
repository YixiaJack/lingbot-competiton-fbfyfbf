#!/bin/bash
# run_case.sh NAME CODE CFG NGPU [--key value ...]  (LB_* / LINGBOT_* / FORGE_* env vars pass through)
R=$SCRATCH/lingbot/devrep_20261114
NAME=$1; CODE=$2; CFG=$3; N=$4; shift 4
OUT=$R/runs/$NAME; mkdir -p "$OUT"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${PYEXTRA:+$PYEXTRA:}$CODE:$R/aux
PORT=$((29500 + RANDOM % 2000))
echo "$(date -Is) NAME=$NAME CODE=$CODE CFG=$CFG N=$N ARGS=$* ENV=$(env | grep -E '^(LB_|LINGBOT_|FORGE_)' | sort | tr '\n' ' ')" >> "$OUT/launch.txt"
cd "$OUT"
timeout "${CASE_TIMEOUT:-40m}" bash $SCRATCH/lingbot/LingBotRoboTwin/scripts/hpc/run_in_env.sh \
  torchrun --nproc-per-node "$N" --master-port "$PORT" "$CODE/tasks/vla/train_lingbotvla.py" "$CFG" --train.output_dir "$OUT" "$@" \
  > "$OUT/train.log" 2>&1
rc=$?; echo "$(date -Is) exit $rc" >> "$OUT/launch.txt"; echo "$NAME exit $rc"
