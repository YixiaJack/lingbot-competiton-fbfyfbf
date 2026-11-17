#!/usr/bin/env bash
# Convert all 50 tasks in one CPU job with a light host venv (lerobot 0.4.2 + CPU torch), so conversion does
# not wait for the overlay. Same converter as convert.sbatch; finished tasks (.done) are skipped.
#   scripts/hpc/submit.sh --cpu scripts/hpc/convert_all.sbatch
#SBATCH --job-name=lb_convert_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/lingbot/logs/convert_all_%j.log
set -euo pipefail
cd "${WORK_DIR:-/scratch/${USER}/lingbot/LingBotRoboTwin}"
source scripts/hpc/env.sh
VENV="${PROJ}/.convvenv"
if [[ ! -f "${VENV}/.ok" ]]; then
  python3 -m venv "${VENV}"
  "${VENV}/bin/pip" install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
  # --no-deps like upstream's env script (lerobot's robot extras pull evdev, which needs Python headers)
  "${VENV}/bin/pip" install -q --no-deps "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz"
  "${VENV}/bin/pip" install -q h5py opencv-python-headless "datasets==3.6.0" "huggingface-hub==0.34.3" "pandas==2.3.1" "pyarrow==21.0.0" \
    pillow jsonlines packaging "av==15.0.0" imageio termcolor einops deepdiff draccus accelerate "numpy<2" tqdm
  "${VENV}/bin/python" -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('lerobot ok')"
  touch "${VENV}/.ok"
fi
convert_one() {
  local t="$1" out="${DATA_DIR}/lerobot/robotwin_clean/$1-aloha-agilex_clean_50-50"
  [[ -f "${out}/.done" ]] && return 0
  "${VENV}/bin/python" src/convert_robotwin_to_lerobot.py --raw_dir "${DATA_DIR}/raw/${t}/aloha-agilex_clean_50" \
    --out_dir "${out}" > "${LOG_DIR}/convert_${t}.log" 2>&1 && touch "${out}/.done" && echo "ok ${t} $(tail -1 "${LOG_DIR}/convert_${t}.log")" \
    || echo "FAILED ${t}: $(tail -3 "${LOG_DIR}/convert_${t}.log")"
}
export -f convert_one; export VENV DATA_DIR LOG_DIR
python3 -c "import json;print('\n'.join(json.load(open('configs/tasks50.json'))))" | xargs -P 8 -I{} bash -c 'convert_one {}'
echo "=== converted: $(ls ${DATA_DIR}/lerobot/robotwin_clean/*/.done | wc -l)/50 $(date)"
du -sh "${DATA_DIR}/lerobot/robotwin_clean"
