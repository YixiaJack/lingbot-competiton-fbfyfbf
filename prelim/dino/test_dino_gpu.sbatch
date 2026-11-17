#!/usr/bin/env bash
# GPU smoke test of the DINOv2 fusion on the DINO dev checkout (never the live lingbot-vla-v2):
#   1. model_path = BASE hf_ckpt + DINOv2 (zero-init projection), laid out as a deploy-style run dir under dev/
#   2. 20 training steps with the full continuation recipe (aug_strong, sampling, ...) -> checkpoint
#   3. dino/test_dino_gpu.py: zero-init == base actions, projection is live, trained checkpoint loads
#   scripts/hpc/submit.sh dino/test_dino_gpu.sbatch --gres=gpu:a100:2 --constraint=a100   (twins: same command with h100)
#SBATCH --job-name=lb_dinogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/lingbot/logs/dinogpu_%j.log
set -euo pipefail
cd "${WORK_DIR:-/scratch/${USER}/lingbot/LingBotRoboTwin}"
export LINGBOT_DIR="${PROJ:-/scratch/${USER}/lingbot}/dev/lingbot-vla-v2-dino"
source scripts/hpc/env.sh
BASE_RUN="${BASE_RUN:-strong_scratch24k}"; BASE_STEP="${BASE_STEP:-22000}"
BASE="${RUN_DIR}/${BASE_RUN}/checkpoints/global_step_${BASE_STEP}/hf_ckpt"
DINO_DIR="${CKPT_DIR}/dinov2-with-registers-base"
INIT_RUN="${PROJ}/dev/runs/dino_init_${BASE_RUN}_${BASE_STEP}_${SLURM_JOB_ID}"  # per job: twins never share it
INIT="${INIT_RUN}/checkpoints/global_step_0/hf_ckpt"
# twins on other GPU types (same --job-name): drop the ones still queued now that this one runs (PENDING only)
scancel -t PENDING -u "${USER}" -n "${SLURM_JOB_NAME}" || true
python3 src/patch_upstream.py "${LINGBOT_DIR}" >/dev/null && python3 dino/patch_dino.py "${LINGBOT_DIR}" | tail -1
RUN() { scripts/hpc/run_in_env.sh "$@"; }

echo "=== 1. init model_path ${INIT}"
if [[ ! -f "${INIT}/model.safetensors.index.json" ]]; then
  RUN python dino/make_dino_init.py --base "${BASE}" --dino "${DINO_DIR}" --out "${INIT}"
fi
RUN python - "${RUN_DIR}/${BASE_RUN}/lingbotvla_cli.yaml" "${INIT_RUN}/lingbotvla_cli.yaml" "${DINO_DIR}" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg["train"].update(dino_fusion_path=sys.argv[3], dino_fusion_input_size=224, dino_fusion_grid=8)
yaml.safe_dump(cfg, open(sys.argv[2], "w"), sort_keys=False)
EOF

echo "=== 2. training smoke"
export PYTHONPATH="${LINGBOT_DIR}:${WORK_DIR}/src" LINGBOT_AUG_CONFIG="${WORK_DIR}/configs/aug_strong.json" LINGBOT_SEEN_POOL=1 \
       LINGBOT_STATE_NOISE=0.01 LINGBOT_VISION_LR_SCALE=0.1 LINGBOT_TASK_SAMPLING=uniform LINGBOT_FRAME_WEIGHTS=1 \
       LINGBOT_DINO_LR_SCALE=10
SMOKE=dino_smoke_${SLURM_JOB_ID}
RUN_DIR="${PROJ}/dev/runs" RUN_NAME=${SMOKE} MAX_STEPS=20 SAVE_STEPS=20 MICRO_BS=2 GLOBAL_BS=4 \
  TRAIN_ARGS="--model.model_path ${INIT} --train.dino_fusion_path ${DINO_DIR} --data.image_augment true --train.lr 2e-5 --train.lr_min 5e-6" \
  bash scripts/hpc/train.sbatch
TRAINED="${PROJ}/dev/runs/${SMOKE}/checkpoints/global_step_20/hf_ckpt"
grep -c "dino_fusion" "${TRAINED}/model.safetensors.index.json" | sed 's/^/    dino keys in trained hf_ckpt index: /'

echo "=== 3. inference-path checks"
cd "${LINGBOT_DIR}"
"${WORK_DIR}/scripts/hpc/run_in_env.sh" python "${WORK_DIR}/dino/test_dino_gpu.py" --base "${BASE}" --dino_init "${INIT}" \
  --trained "${TRAINED}"
