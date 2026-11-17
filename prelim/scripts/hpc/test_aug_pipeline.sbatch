#!/usr/bin/env bash
# CPU job: apply src/patch_upstream.py to a fresh COPY of lingbot-vla-v2 (the live checkout is not touched), run
# src/test_aug_pipeline.py against it, and render an augmentation preview.
#   AUG=configs/aug_strong.json scripts/hpc/submit.sh --cpu scripts/hpc/test_aug_pipeline.sbatch
#SBATCH --job-name=lb_augtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/lingbot/logs/augtest_%j.log
set -euo pipefail
cd "${WORK_DIR:-/scratch/${USER}/lingbot/LingBotRoboTwin}"
source scripts/hpc/env.sh
AUG="${AUG:-configs/aug_strong.json}"
T="${SCRATCH}/TMPDIR/augtest_${SLURM_JOB_ID}"
mkdir -p "${T}"
rsync -a --exclude .git --exclude __pycache__ "${LINGBOT_DIR}/" "${T}/lingbot-vla-v2/"
python3 src/patch_upstream.py "${T}/lingbot-vla-v2"
python3 src/patch_upstream.py "${T}/lingbot-vla-v2" | grep -c "already patched"
apptainer exec --overlay "${OVERLAY}:ro" "${IMAGE}" bash -c "source /ext3/env.sh >/dev/null 2>&1; conda activate lingbotvla; \
  PYTHONPATH=${T}/lingbot-vla-v2:${WORK_DIR}/src python src/test_aug_pipeline.py ${AUG} ${T}/fake"
apptainer exec "${IMAGE}" bash -c "source ${PROJ}/miniforge3/etc/profile.d/conda.sh && conda activate RoboTwin && \
  LINGBOT_AUG_CONFIG=${AUG} python src/preview_aug.py --raw_root ${DATA_DIR}/raw \
  --tasks lift_pot click_bell hanging_mug stack_blocks_three open_microwave --out ${T}/preview.jpg"
echo "=== preview ${T}/preview.jpg"
