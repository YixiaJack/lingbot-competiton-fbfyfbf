#!/usr/bin/env bash
# 一键评测：加载提交的 checkpoint，在 demo_clean 和 demo_randomized 两个 setting 下评测全部 50 个任务，并生成 results.json。
# 单机多卡，每张卡同时评一个任务；每个任务 = 一个 LingBot-VLA 推理服务 + 一个 RoboTwin 评测客户端（src/eval_one_task.sh）。
#
#   export LINGBOT_DIR=... ROBOTWIN_DIR=... CKPT_DIR=...
#   bash eval.sh                       # 官方协议：每个任务 100 次
#   EVAL_TEST_NUM=10 bash eval.sh      # 本次提交 results.json 的口径：每个任务 10 次
#
# 每个 episode 的场景只由种子决定，种子从 100000 起依次使用，所以 EVAL_TEST_NUM=10 的结果与 100 次评测的前 10 次完全相同。
#
# LINGBOT_DIR   `DEST=... bash src/setup_lingbot.sh` 生成的代码目录（推理只需要它，不需要训练器补丁）
# ROBOTWIN_DIR  RoboTwin @ 13c3c47 及其资源（src/prepare_data.sh 下载）
# CKPT_DIR      基座模型目录（需要其中的 Qwen3-VL-4B-Instruct；src/prepare_data.sh 下载）
# 可选：CKPT      默认为本提交包的 03_模型权重/checkpoint/v3uni_noon/checkpoints/global_step_5000/hf_ckpt
#       OUT       输出目录，默认 ./eval_out
#       SETTINGS  默认 "demo_clean demo_randomized"
#       CONDA_SH / SIM_CONDA_SH  含 conda 环境 lingbotvla / RoboTwin 的 conda.sh（默认取 `conda info --base`）
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LINGBOT_DIR:?}" "${ROBOTWIN_DIR:?}" "${CKPT_DIR:?}"
CKPT="${CKPT:-${HERE}/../03_模型权重/checkpoint/v3uni_noon/checkpoints/global_step_5000/hf_ckpt}"
OUT="${OUT:-${PWD}/eval_out}"
SETTINGS="${SETTINGS:-demo_clean demo_randomized}"
export EVAL_TEST_NUM="${EVAL_TEST_NUM:-100}"
[[ -s "${CKPT}/model.safetensors.index.json" ]] || { echo "找不到 checkpoint: ${CKPT}" >&2; exit 1; }
mkdir -p "${OUT}"

# checkpoint 自带的 lingbotvla_cli.yaml 里是训练集群上的绝对路径：在 $OUT/_staged 生成路径本地化的副本
# （权重只做软链接），归一化统计固定为本目录的 clean-only 文件并校验 SHA-256，不一致就拒绝评测
STAGED=$(python3 "${HERE}/src/stage_ckpt.py" --ckpt "${CKPT}" --ckpt_dir "${CKPT_DIR}" --stage "${OUT}/_staged") || exit 1
echo "serving ${STAGED}  (${EVAL_TEST_NUM} episodes per task)"
export WORK_DIR="${HERE}" LINGBOT_DIR ROBOTWIN_DIR
NGPU=$(nvidia-smi -L | wc -l)
mapfile -t TASKS < <(python3 -c "import json;print('\n'.join(json.load(open('${HERE}/configs/tasks50.json'))))")

done_already() {  # <setting> <task> -> 0 if its .result already has EVAL_TEST_NUM episodes
  local f="${OUT}/$1/$2.result"
  [[ -f "${f}" ]] && (( $(awk '{print $2}' "${f}") >= EVAL_TEST_NUM ))
}
run_gpu() {  # <setting> <gpu>
  local s=$1 g=$2 i t
  for ((i=g; i<${#TASKS[@]}; i+=NGPU)); do
    t=${TASKS[$i]}
    done_already "${s}" "${t}" && continue
    CUDA_VISIBLE_DEVICES=${g} bash "${HERE}/src/eval_one_task.sh" "${STAGED}" "${t}" "${s}" "${OUT}/${s}" $((9400 + g))
  done
}
for s in ${SETTINGS}; do
  for ((g=0; g<NGPU; g++)); do run_gpu "${s}" "${g}" & done
  wait
  echo "${s}: $(ls "${OUT}/${s}"/*.result 2>/dev/null | wc -l)/50 个任务完成"
done

# 汇总成官方模板格式：每个 (setting, 任务) 取前 EVAL_TEST_NUM 个 episode
python3 "${HERE}/src/make_results_first_n.py" --eval_dir "${OUT}" --episodes "${EVAL_TEST_NUM}" \
  --team_id "${TEAM_ID:-<team_id>}" --out "${OUT}/results.json" && echo "results: ${OUT}/results.json"
