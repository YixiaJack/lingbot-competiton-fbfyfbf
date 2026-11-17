#!/usr/bin/env bash
# 下载基座模型、RoboTwin 代码与资源、50 个任务的 aloha-agilex_clean_50 数据，并转换成训练用的 LeRobot 数据集。
# 只下载 clean 数据，不下载、不使用任何 randomized 数据。可重复运行，已完成的步骤会跳过。
#
#   CKPT_DIR=/path/to/ckpts DATA_DIR=/path/to/data ROBOTWIN_DIR=/path/to/RoboTwin bash src/prepare_data.sh
#
# 需要：git、unzip、Hugging Face CLI（`pip install "huggingface_hub>=0.34"`，提供 `hf` 命令）；
# 转换用的 Python 需要 lerobot 0.4.2、h5py、opencv、av（conda 环境 lingbotvla 即可，可用 PYTHON=... 指定）。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CKPT_DIR:?}" "${DATA_DIR:?}" "${ROBOTWIN_DIR:?}"
PY="${PYTHON:-python3}"
ROBOTWIN_COMMIT=13c3c47ff4312dd62484bcd51be034af55c062d1
mapfile -t TASKS < <(python3 -c "import json;print('\n'.join(json.load(open('${HERE}/../configs/tasks50.json'))))")

# ---------- RoboTwin（评测用仿真器），LingBot 文档指定的版本 ----------
[[ -d "${ROBOTWIN_DIR}/.git" ]] || git clone https://github.com/RoboTwin-Platform/RoboTwin.git "${ROBOTWIN_DIR}"
git -C "${ROBOTWIN_DIR}" checkout -q "${ROBOTWIN_COMMIT}"

# ---------- 模型权重 ----------
hf download robbyant/lingbot-vla-v2-6b --local-dir "${CKPT_DIR}/lingbot-vla-v2-6b" --exclude "assets/*"   # 含 depth/、dino_video/ 教师
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir "${CKPT_DIR}/Qwen3-VL-4B-Instruct"
hf download Ruicheng/moge-2-vitb-normal --local-dir "${CKPT_DIR}/moge-2-vitb-normal"

# ---------- RoboTwin 仿真资源 ----------
hf download TianxingChen/RoboTwin2.0 background_texture.zip embodiments.zip objects.zip \
  --repo-type dataset --local-dir "${ROBOTWIN_DIR}/assets"
for z in background_texture embodiments objects; do
  [[ -d "${ROBOTWIN_DIR}/assets/${z}" ]] || (cd "${ROBOTWIN_DIR}/assets" && unzip -q "${z}.zip" && rm -f "${z}.zip")
done

# ---------- 演示数据：每个任务只取 aloha-agilex_clean_50 ----------
for t in "${TASKS[@]}"; do
  hf download TianxingChen/RoboTwin2.0 --repo-type dataset --local-dir "${DATA_DIR}/raw_zip" \
    --include "dataset/${t}/aloha-agilex_clean_50.zip" >/dev/null
  dst="${DATA_DIR}/raw/${t}"
  if [[ ! -f "${dst}/.unzipped" ]]; then
    mkdir -p "${dst}" && unzip -q -o "${DATA_DIR}/raw_zip/dataset/${t}/aloha-agilex_clean_50.zip" -d "${dst}" && touch "${dst}/.unzipped"
  fi
done

# ---------- 转换为 LeRobot v3.0（视频格式），见 convert_robotwin_to_lerobot.py ----------
for t in "${TASKS[@]}"; do
  out="${DATA_DIR}/lerobot/robotwin_clean/${t}-aloha-agilex_clean_50-50"
  [[ -f "${out}/.done" ]] && continue
  "${PY}" "${HERE}/convert_robotwin_to_lerobot.py" --raw_dir "${DATA_DIR}/raw/${t}/aloha-agilex_clean_50" --out_dir "${out}" \
    && touch "${out}/.done"
done
echo "converted: $(ls "${DATA_DIR}"/lerobot/robotwin_clean/*/.done | wc -l)/50"

# ---------- 训练期用到的两份元数据（只读 clean 演示本身） ----------
"${PY}" "${HERE}/make_seen_pool.py" --raw_root "${DATA_DIR}/raw" --lerobot_root "${DATA_DIR}/lerobot/robotwin_clean"   # seen 指令池
"${PY}" "${HERE}/make_frame_weights.py" --lerobot_root "${DATA_DIR}/lerobot/robotwin_clean"                         # 帧权重（默认 0.3 / 1）
echo "done: ${DATA_DIR}/lerobot/robotwin_clean"
