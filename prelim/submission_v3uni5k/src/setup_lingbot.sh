#!/usr/bin/env bash
# 准备 LingBot-VLA 2.0 代码：官方 lingbot-vla-v2 @ be969b8，再打上本目录的补丁。可重复运行。
#
#   DEST=/path/to/lingbot-vla-v2        bash src/setup_lingbot.sh                  # 评测 + 第 1、2 段训练
#   DEST=/path/to/lingbot-vla-v2-stage3 TRAINER_PATCH=1 bash src/setup_lingbot.sh  # 第 3 段训练
#
# 补丁（说明见 README「对官方代码的改动」）：
#   src/patch_upstream.py   方法改动，结果等于 src/patches/method.diff：训练期白名单增广、seen 指令池替换、本体状态噪声、
#                           任务均匀采样与帧权重、视觉塔学习率倍率。全部由环境变量开关，不设置时与官方代码行为一致；
#                           推理路径不受影响。
#   src/patches/trainer_stage3.diff（TRAINER_PATCH=1）  第 3 段训练实际使用的训练器改动：每个 rank 独立的随机流、
#                           续训时沿用配置中的学习率、不同卡数之间续训、按时间存档与清理旧 checkpoint。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT=be969b8fd117fb70550c5d4bf4bc328211b5b1b6
DEST="${DEST:-${PWD}/lingbot-vla-v2}"
[[ -d "${DEST}/.git" ]] || git clone https://github.com/robbyant/lingbot-vla-v2.git "${DEST}"
git -C "${DEST}" checkout -q "${COMMIT}"
python3 "${HERE}/patch_upstream.py" "${DEST}"
if [[ "${TRAINER_PATCH:-0}" == 1 ]]; then
  if grep -q "_lb_reseed" "${DEST}/tasks/vla/train_lingbotvla.py"; then
    echo "trainer patch already applied"
  else
    git -C "${DEST}" apply "${HERE}/patches/trainer_stage3.diff"
  fi
fi
git -C "${DEST}" diff --stat
echo "ready: ${DEST}"
