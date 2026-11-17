#!/usr/bin/env bash
# 准备 LingBot-VLA 2.0 代码：官方 lingbot-vla-v2 @ be969b8，再打上本目录的补丁。可重复运行。
#
#   DEST=/path/to/lingbot-vla-v2 bash src/setup_lingbot.sh      # 训练（两段）与评测共用
#
# 补丁（说明见 README「对官方代码的改动」）：
#   src/patch_upstream.py   方法改动，结果等于 src/patches/method.diff：训练期白名单增广、seen 指令池替换、本体状态噪声、
#                           任务均匀采样与帧权重、视觉塔学习率倍率。全部由环境变量开关，不设置时与官方代码行为一致；
#                           推理路径不受影响。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT=be969b8fd117fb70550c5d4bf4bc328211b5b1b6
DEST="${DEST:-${PWD}/lingbot-vla-v2}"
[[ -d "${DEST}/.git" ]] || git clone https://github.com/robbyant/lingbot-vla-v2.git "${DEST}"
git -C "${DEST}" checkout -q "${COMMIT}"
python3 "${HERE}/patch_upstream.py" "${DEST}"
git -C "${DEST}" diff --stat
echo "ready: ${DEST}"
