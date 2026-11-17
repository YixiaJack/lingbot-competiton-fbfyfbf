#!/usr/bin/env bash
# CPU job: patch the DINO dev checkout (src/patch_upstream.py + dino/patch_dino.py) and run dino/test_dino_cpu.py.
#   scripts/hpc/submit.sh --cpu dino/test_dino_cpu.sbatch
#SBATCH --job-name=lb_dinotest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/lingbot/logs/dinotest_%j.log
set -euo pipefail
cd "${WORK_DIR:-/scratch/${USER}/lingbot/LingBotRoboTwin}"
source scripts/hpc/env.sh
DEV="${DINO_DEV:-${PROJ}/dev/lingbot-vla-v2-dino}"
python3 src/patch_upstream.py "${DEV}"
python3 dino/patch_dino.py "${DEV}"
git -C "${DEV}" diff --stat | tail -1
apptainer exec --overlay "${OVERLAY}:ro" "${IMAGE}" bash -c "source /ext3/env.sh >/dev/null 2>&1; conda activate lingbotvla; \
  python -c 'import h5py, cv2' 2>/dev/null || pip install --quiet --target ${TMPDIR}/dinotest_py h5py; \
  PYTHONPATH=${DEV}:${WORK_DIR}/dino:${TMPDIR}/dinotest_py python dino/test_dino_cpu.py --qwen ${CKPT_DIR}/Qwen3-VL-4B-Instruct \
    --dino ${CKPT_DIR}/dinov2-with-registers-base --frame_h5 ${DATA_DIR}/raw/lift_pot/aloha-agilex_clean_50/data/episode0.hdf5"
