#!/usr/bin/env bash
# 一键训练（单机多卡，torchrun），复现本次提交的 checkpoint：v3uni_noon / global_step_5000。
# 只使用 RoboTwin 2.0 Aloha-AgileX 50 个任务的 aloha-agilex_clean_50 数据，不使用任何 randomized 数据。
#
# 提交的 checkpoint 是三段接力训练（共 35,000 步）。每段的超参数、增广配置、随机种子都内置在下面的 STAGE 预设里：
#
#   export LINGBOT_DIR=... LINGBOT_DIR_STAGE3=... CKPT_DIR=... DATA_DIR=... RUNS_DIR=...
#   STAGE=1   bash train.sh   # 从基座训练：学习率日程按 24,000 步，到 global_step_22000 自动停
#   STAGE=2   bash train.sh   # 从第 1 段 global_step_22000 续训 8,000 步
#   STAGE=3   bash train.sh   # 从第 2 段 global_step_8000 续训：学习率日程按 15,000 步，到 global_step_5000 自动停
#   STAGE=all bash train.sh   # 依次跑完三段
#
# LINGBOT_DIR         第 1、2 段的训练代码：`DEST=... bash src/setup_lingbot.sh`
# LINGBOT_DIR_STAGE3  第 3 段的训练代码：`DEST=... TRAINER_PATCH=1 bash src/setup_lingbot.sh`（多一份训练器补丁，见 README）
# CKPT_DIR            基座模型目录（lingbot-vla-v2-6b、Qwen3-VL-4B-Instruct、moge-2-vitb-normal），见 src/prepare_data.sh
# DATA_DIR            数据根目录（src/prepare_data.sh 的输出，训练读 $DATA_DIR/lerobot/robotwin_clean/ 下 50 个数据集）
# RUNS_DIR            输出根目录：$RUNS_DIR/stage1 | stage2 | stage3
#
# 全局 batch 固定 128，卡数不同时由梯度累积补齐。显存不够时（例如 80 GB 卡）可以：
#   MICRO_BS=8 EXTRA_ARGS="--train.enable_gradient_checkpointing true --train.enable_full_shard true" STAGE=... bash train.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${STAGE:?STAGE=1|2|3|all}" "${LINGBOT_DIR:?}" "${CKPT_DIR:?}" "${DATA_DIR:?}" "${RUNS_DIR:?}"
LINGBOT_DIR_STAGE3="${LINGBOT_DIR_STAGE3:-${LINGBOT_DIR}-stage3}"
NGPU=$(nvidia-smi -L | wc -l)
OLD='\$SCRATCH/lingbot'   # 配置文件里记录的是我们集群上的路径，下面改写成上面的变量
CUR_PG=""                     # torchrun 在独立进程组里跑；Ctrl+C 时连同所有 rank 一起停掉
trap '[[ -n "${CUR_PG}" ]] && kill -TERM -- "-${CUR_PG}" 2>/dev/null; exit 130' INT TERM

# 训练期改动全部由环境变量开关（src/patch_upstream.py），三段共用的部分：
export PYTHONPATH="${HERE}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LINGBOT_SEEN_POOL=1 LINGBOT_STATE_NOISE=0.01 LINGBOT_VISION_LR_SCALE=0.1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

hf_files_ok() {  # <hf_ckpt> -> 0 if config, index and every shard named in the index are there
  local h=$1
  [[ -s "${h}/config.json" && -s "${h}/model.safetensors.index.json" ]] || return 1
  python3 - "${h}" <<'PY'
import json, os, sys
h = sys.argv[1]
shards = set(json.load(open(os.path.join(h, "model.safetensors.index.json")))["weight_map"].values())
sys.exit(0 if all(os.path.isfile(os.path.join(h, s)) and os.path.getsize(os.path.join(h, s)) > 0 for s in shards) else 1)
PY
}
hf_complete() {  # <hf_ckpt> -> 0 if hf_files_ok and nothing was written for 3 min
  hf_files_ok "$1" && [[ -z "$(find "$1" -newermt '3 minutes ago' -print -quit)" ]]
}

# run <lingbot_dir> <output_dir> <micro_bs> <max_steps> <save_steps> <stop_at|""> <extra train args...>
run() {
  local code=$1 out=$2 micro=$3 steps=$4 save=$5 stop=$6; shift 6
  local cfg="${out}/_cfg"
  mkdir -p "${cfg}/robot_configs" "${cfg}/norm_stats"
  sub() { sed -e "s#${OLD}/ckpts#${CKPT_DIR}#g" -e "s#${OLD}/data/lerobot/robotwin_clean#${DATA_DIR}/lerobot/robotwin_clean#g" \
              -e "s#${OLD}/LingBotRoboTwin/configs#${cfg}#g" -e "s#${OLD}/runs/robotwin_clean#${out}#g" "$1"; }
  sub "${HERE}/configs/robotwin_clean.yaml" > "${cfg}/robotwin_clean.yaml"
  sub "${HERE}/configs/robotwin_clean.txt" > "${cfg}/robotwin_clean.txt"
  sub "${HERE}/configs/robot_configs/robotwin_clean.yaml" > "${cfg}/robot_configs/robotwin_clean.yaml"
  cp "${HERE}/configs/norm_stats/robotwin_clean.json" "${cfg}/norm_stats/"
  if [[ -n "${stop}" ]] && hf_complete "${out}/checkpoints/global_step_${stop}/hf_ckpt"; then
    echo "global_step_${stop} 已存在，跳过"; return 0
  fi
  # 上次中断在写 global_step_${stop} 的过程中：训练器会从这个半成品续训、越过 ${stop}，下面的监视器永远等不到它写完
  # （第 3 段的训练器补丁还会在下一次存档时把它删掉）。挪开（不删），从上一个 checkpoint 续训
  local tgt="${out}/checkpoints/global_step_${stop}"
  if [[ -n "${stop}" && -d "${tgt}" ]] && ! hf_files_ok "${tgt}/hf_ckpt"; then
    mv "${tgt}" "${tgt}.incomplete_$(date +%s)"
    echo "global_step_${stop} 没写完，已挪到 ${tgt}.incomplete_*（确认无用后可手动删除），从上一个 checkpoint 续训"
  fi
  echo "=== $(date '+%F %T') ${code} -> ${out}  max_steps=${steps} stop_at=${stop:-max} $*"
  ( cd "${code}" && exec setsid torchrun --nnodes=1 --nproc-per-node "${NGPU}" --master-port="${MASTER_PORT:-29531}" \
      tasks/vla/train_lingbotvla.py "${cfg}/robotwin_clean.yaml" \
      --data.norm_stats_file "${cfg}/norm_stats/robotwin_clean.json" --train.output_dir "${out}" \
      --train.micro_batch_size "${micro}" --train.global_batch_size 128 \
      --train.max_steps "${steps}" --train.save_steps "${save}" "$@" ${EXTRA_ARGS:-} ) &
  local pg=$! watcher=""; CUR_PG=${pg}
  if [[ -n "${stop}" ]]; then  # 学习率日程按 max_steps 计算，只在指定的 checkpoint 写完后停止
    ( while kill -0 "${pg}" 2>/dev/null; do
        if hf_complete "${out}/checkpoints/global_step_${stop}/hf_ckpt"; then
          echo "global_step_${stop} 已写完，停止本段"; kill -TERM -- "-${pg}" 2>/dev/null; break
        fi
        sleep 60
      done ) &
    watcher=$!
  fi
  local rc=0; wait "${pg}" || rc=$?
  [[ -n "${watcher}" ]] && { kill "${watcher}" 2>/dev/null || true; wait "${watcher}" 2>/dev/null || true; }
  if [[ -n "${stop}" ]]; then hf_complete "${out}/checkpoints/global_step_${stop}/hf_ckpt"; else return "${rc}"; fi
}

stage1() {  # 22,000 步（日程 24,000），Muon，lr 1e-4 -> 5e-5 余弦、无 warmup（configs/robotwin_clean.yaml 默认值），seed 42
  LINGBOT_AUG_CONFIG="${HERE}/configs/aug_whitelist.json" \
  run "${LINGBOT_DIR}" "${RUNS_DIR}/stage1" "${MICRO_BS:-32}" 24000 2000 22000 \
      --data.image_augment true
}
stage2() {  # 8,000 步，lr 2e-5 -> 5e-6 余弦，warmup 3%，seed 42；任务均匀采样 + 低动作帧降权
  LINGBOT_AUG_CONFIG="${HERE}/configs/aug_strong.json" LINGBOT_TASK_SAMPLING=uniform LINGBOT_FRAME_WEIGHTS=1 \
  run "${LINGBOT_DIR}" "${RUNS_DIR}/stage2" "${MICRO_BS:-16}" 8000 1000 "" \
      --model.model_path "${RUNS_DIR}/stage1/checkpoints/global_step_22000/hf_ckpt" \
      --data.image_augment true --train.lr 2e-5 --train.lr_min 5e-6 --train.lr_warmup_ratio 0.03
}
stage3() {  # 日程 15,000 步，lr 5e-6 -> 2e-6 余弦（decay_ratio 0.75），warmup 3%；提交的是 global_step_5000
  # 重跑时：存完 5000 后训练器补丁已清掉 4500，第一步会找不到它而从 5000 续训，并在下一次存档时删掉没有 keep 的 5000
  if hf_complete "${RUNS_DIR}/stage3/checkpoints/global_step_5000/hf_ckpt"; then
    touch "${RUNS_DIR}/stage3/checkpoints/global_step_5000/keep"; echo "global_step_5000 已存在，跳过"; return 0
  fi
  # 按时间存档只是为抢占队列准备的续训点（无 hf_ckpt），单机运行时关掉，保证第二步正好从 4500 续训
  local args=(--model.model_path "${RUNS_DIR}/stage2/checkpoints/global_step_8000/hf_ckpt"
              --data.image_augment true --train.lr 5e-6 --train.lr_min 2e-6
              --train.lr_decay_ratio 0.75 --train.lr_warmup_ratio 0.03)
  grep -q "_lb_reseed" "${LINGBOT_DIR_STAGE3}/tasks/vla/train_lingbotvla.py" \
    || { echo "LINGBOT_DIR_STAGE3 缺训练器补丁：DEST=${LINGBOT_DIR_STAGE3} TRAINER_PATCH=1 bash src/setup_lingbot.sh" >&2; return 1; }
  export LINGBOT_AUG_CONFIG="${HERE}/configs/aug_strong_v3.json" LINGBOT_TASK_SAMPLING=uniform LINGBOT_FRAME_WEIGHTS=1 \
         LINGBOT_TIME_SAVE_MIN=0
  # 1-4500 步：主线（seed 42）
  run "${LINGBOT_DIR_STAGE3}" "${RUNS_DIR}/stage3" "${MICRO_BS:-16}" 15000 500 4500 "${args[@]}" --train.seed 42
  # 4501-5000 步：B200 分支接手主线的 global_step_4500，以 seed 43 续训（同一输出目录，自动从 4500 续训）
  run "${LINGBOT_DIR_STAGE3}" "${RUNS_DIR}/stage3" "${MICRO_BS:-16}" 15000 500 5000 "${args[@]}" --train.seed 43
  touch "${RUNS_DIR}/stage3/checkpoints/global_step_5000/keep"   # 训练器补丁清理旧 checkpoint 时保留它
  echo "提交的 checkpoint: ${RUNS_DIR}/stage3/checkpoints/global_step_5000/hf_ckpt"
}

case "${STAGE}" in
  1) stage1 ;; 2) stage2 ;; 3) stage3 ;;
  all) stage1 && stage2 && stage3 ;;
  *) echo "STAGE 应为 1 / 2 / 3 / all" >&2; exit 2 ;;
esac
