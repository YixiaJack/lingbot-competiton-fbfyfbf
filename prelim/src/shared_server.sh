#!/usr/bin/env bash
# Policy server shared by the slots of one eval_pool.sbatch job (SHARED_SERVER=1): the same model and flags as the
# per-item server in src/eval_one_task.sh, through src/shared_policy_server.py (per-connection inference state).
# Runs INSIDE the overlay container:  scripts/hpc/run_in_env.sh bash src/shared_server.sh <hf_ckpt> <port>
set -uo pipefail
CKPT="$1"; PORT="$2"
: "${LINGBOT_DIR:?}" "${WORK_DIR:?}"
source "${CONDA_SH:-/ext3/miniforge3/etc/profile.d/conda.sh}"
conda activate lingbotvla
cd "${LINGBOT_DIR}"
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 exec python "${WORK_DIR}/src/shared_policy_server.py" \
  --model_path "${CKPT}" --use_length 50 --use_bf16 "${USE_BF16:-False}" --use_fp32 "${USE_FP32:-True}" \
  --use_compile True --port "${PORT}"
