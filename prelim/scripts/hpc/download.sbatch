#!/usr/bin/env bash
# Fetch code, weights, RoboTwin assets and the 50 aloha-agilex CLEAN demo zips (no randomized data).
#   scripts/hpc/submit.sh --cpu scripts/hpc/download.sbatch
# Re-running is safe: every step skips what already exists.
#SBATCH --job-name=lb_download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/lingbot/logs/download_%j.log
set -euo pipefail
source "${WORK_DIR:-/scratch/${USER}/lingbot/LingBotRoboTwin}/scripts/hpc/env.sh"
LINGBOT_COMMIT=be969b8fd117fb70550c5d4bf4bc328211b5b1b6
ROBOTWIN_COMMIT=13c3c47ff4312dd62484bcd51be034af55c062d1
echo "=== download on $(hostname) $(date)"

# ---------- code ----------
[[ -d "${LINGBOT_DIR}/.git" ]] || git clone https://github.com/robbyant/lingbot-vla-v2.git "${LINGBOT_DIR}"
git -C "${LINGBOT_DIR}" checkout -q "${LINGBOT_COMMIT}"
[[ -d "${ROBOTWIN_DIR}/.git" ]] || git clone https://github.com/RoboTwin-Platform/RoboTwin.git "${ROBOTWIN_DIR}"
git -C "${ROBOTWIN_DIR}" checkout -q "${ROBOTWIN_COMMIT}"

# ---------- hf CLI in a tiny host venv ----------
VENV="${PROJ}/.dlvenv"
[[ -x "${VENV}/bin/hf" ]] || { python3 -m venv "${VENV}"; "${VENV}/bin/pip" install -q "huggingface_hub[hf_transfer]>=0.34"; }
HF="${VENV}/bin/hf"

# ---------- weights ----------
"${HF}" download robbyant/lingbot-vla-v2-6b --local-dir "${CKPT_DIR}/lingbot-vla-v2-6b" --exclude "assets/*"
"${HF}" download Qwen/Qwen3-VL-4B-Instruct --local-dir "${CKPT_DIR}/Qwen3-VL-4B-Instruct"
"${HF}" download Ruicheng/moge-2-vitb-normal --local-dir "${CKPT_DIR}/moge-2-vitb-normal"

# ---------- RoboTwin simulation assets ----------
"${HF}" download TianxingChen/RoboTwin2.0 background_texture.zip embodiments.zip objects.zip \
  --repo-type dataset --local-dir "${ROBOTWIN_DIR}/assets"
for z in background_texture embodiments objects; do
  if [[ ! -d "${ROBOTWIN_DIR}/assets/${z}" ]]; then
    (cd "${ROBOTWIN_DIR}/assets" && unzip -q "${z}.zip" && rm -f "${z}.zip")
  fi
done

# ---------- demo data: aloha-agilex_clean_50 only ----------
TASKS=$(python3 -c "import json;print(' '.join(json.load(open('${WORK_DIR}/configs/tasks50.json'))))")
for t in ${TASKS}; do
  "${HF}" download TianxingChen/RoboTwin2.0 --repo-type dataset --local-dir "${DATA_DIR}/raw_zip" \
    --include "dataset/${t}/aloha-agilex_clean_50.zip" >/dev/null
done
for t in ${TASKS}; do
  dst="${DATA_DIR}/raw/${t}"
  if [[ ! -f "${dst}/.unzipped" ]]; then
    mkdir -p "${dst}"
    unzip -q -o "${DATA_DIR}/raw_zip/dataset/${t}/aloha-agilex_clean_50.zip" -d "${dst}"
    touch "${dst}/.unzipped"
  fi
done
echo "--- sizes"; du -sh "${CKPT_DIR}"/* "${ROBOTWIN_DIR}/assets" "${DATA_DIR}/raw" 2>/dev/null
echo "=== done $(date)"
