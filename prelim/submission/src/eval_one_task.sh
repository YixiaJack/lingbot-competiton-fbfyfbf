#!/usr/bin/env bash
# Evaluate one RoboTwin task in one setting with one GPU: start a LingBot-VLA inference server, run the
# RoboTwin eval client against it, write <out_dir>/<task>.result ("successes attempts").
# This is upstream experiment/robotwin/start_robotwin_infer_and_eval.sh reduced to a single (task, slot),
# with the same server / client commands and arguments, so tasks can run as independent (preemptible) jobs.
# Needs two conda envs: `lingbotvla` (inference server) and `RoboTwin` (simulator); see README.
#
#   eval_one_task.sh <hf_ckpt> <task_name> <demo_clean|demo_randomized> <out_dir> [port]
# Env: EVAL_TEST_NUM (default 100, the official count), ROBO_NAME (default robotwin_clean),
#      INSTRUCTION_TYPE (default: seen for demo_clean, unseen for demo_randomized, per rule A.5),
#      USE_BF16/USE_FP32 (default False/True, the release setting), EVAL_VIDEO (default False).
set -uo pipefail
CKPT="$1"; TASK="$2"; TASK_CONFIG="$3"; OUT="$4"
# free port from the OS unless given
PORT="${5:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')}"
: "${LINGBOT_DIR:?}" "${ROBOTWIN_DIR:?}" "${WORK_DIR:?}"
ROBO_NAME="${ROBO_NAME:-robotwin_clean}"
# Instruction pool per setting (competition rule A.5): clean2clean = seen, clean2random = unseen. The LingBot
# launcher's policy/ACT/deploy_policy.yml says unseen for both, so set it explicitly per setting.
case "${TASK_CONFIG}" in
  demo_clean) DEFAULT_INSTRUCTION=seen ;;
  *) DEFAULT_INSTRUCTION=unseen ;;
esac
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-${DEFAULT_INSTRUCTION}}"
CONDA_SH="${CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}"  # conda that has env `lingbotvla`
SIM_CONDA_SH="${SIM_CONDA_SH:-${CONDA_SH}}"                                     # conda that has env `RoboTwin`
mkdir -p "${OUT}/inference_logs" "${OUT}/eval_logs" "${OUT}/eval_results"
INF_LOG="${OUT}/inference_logs/${TASK}.log"; EVAL_LOG="${OUT}/eval_logs/${TASK}.log"

# --- one-time setup that upstream's launcher also does (idempotent) ---
cp -f "${WORK_DIR}/configs/robot_configs/${ROBO_NAME}.yaml" "${LINGBOT_DIR}/configs/robot_configs/${ROBO_NAME}.yaml"
mkdir -p "${ROBOTWIN_DIR}/script/deploy"
for f in __init__.py websocket_client_policy.py msgpack_numpy.py; do
  cp -f "${LINGBOT_DIR}/deploy/${f}" "${ROBOTWIN_DIR}/script/deploy/${f}"
done
# patched client copy: EVAL_TEST_NUM (default 100 = official; the submission used 10) and per-episode resume state
# (EVAL_STATE_FILE) so a preempted + requeued job continues at the next seed. See src/patch_eval_client.py.
python3 "${WORK_DIR}/src/patch_eval_client.py" "${LINGBOT_DIR}/experiment/robotwin/eval_policy_client_lingbotvla.py" \
  "${ROBOTWIN_DIR}/script/eval_policy_client_lingbotvla.py.${TASK}.${TASK_CONFIG}.$$" \
  && mv -f "${ROBOTWIN_DIR}/script/eval_policy_client_lingbotvla.py.${TASK}.${TASK_CONFIG}.$$" \
           "${ROBOTWIN_DIR}/script/eval_policy_client_lingbotvla.py"
mkdir -p "${OUT}/state"
export EVAL_STATE_FILE="${OUT}/state/${TASK}.json"

# --- inference server ---
source "${CONDA_SH}"
cd "${LINGBOT_DIR}"
( conda activate lingbotvla && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 exec python -m deploy.lingbot_vla_v2_policy \
    --model_path "${CKPT}" --use_length 50 --use_bf16 "${USE_BF16:-False}" --use_fp32 "${USE_FP32:-True}" \
    --use_compile True --port "${PORT}" ) > "${INF_LOG}" 2>&1 &
INF_PID=$!
trap 'kill ${INF_PID} ${SIM_PID:-} 2>/dev/null; pkill -P ${INF_PID} 2>/dev/null' EXIT
for i in $(seq 1 180); do  # model load + compile warm-up
  (echo > "/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null && break
  kill -0 ${INF_PID} 2>/dev/null || { echo "inference server died, see ${INF_LOG}"; tail -30 "${INF_LOG}"; exit 1; }
  sleep 5
done
echo "inference server up on ${PORT} ($(date))"

# --- RoboTwin client (same arguments as upstream's launcher) ---
cd "${ROBOTWIN_DIR}"
( source "${SIM_CONDA_SH}" && conda activate RoboTwin \
  && export PYTHONPATH="$(python -c 'import site;print(site.getsitepackages()[0])')${PYTHONPATH:+:$PYTHONPATH}" \
  && PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
     exec python -u script/eval_policy_client_lingbotvla.py --config policy/ACT/deploy_policy.yml --overrides \
       --task_name "${TASK}" --task_config "${TASK_CONFIG}" --train_config_name 0 --seed 0 --policy_name ACT \
       --instruction_type "${INSTRUCTION_TYPE}" \
       --port "${PORT}" --robo_name "${ROBO_NAME}" --video_fps 10 --eval_video_log "${EVAL_VIDEO:-False}" \
       --output_dir "${OUT}/eval_results" ) >> "${EVAL_LOG}" 2>&1 &
SIM_PID=$!
wait ${SIM_PID}; RC=$?

LAST=$(sed 's/\x1b\[[0-9;]*m//g' "${EVAL_LOG}" | grep -aoE 'Success rate: [0-9]+/[0-9]+' | tail -1)
echo "client exit ${RC} | ${TASK} ${TASK_CONFIG} | ${LAST}"
if [[ ${RC} -eq 0 && -n "${LAST}" ]]; then
  echo "${LAST#Success rate: }" | tr '/' ' ' > "${OUT}/${TASK}.result"
fi
exit ${RC}
