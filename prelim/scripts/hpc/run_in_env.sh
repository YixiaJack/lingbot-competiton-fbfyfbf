#!/usr/bin/env bash
# Run a command inside the project's python environment, wherever it lives.
#   scripts/hpc/run_in_env.sh python -c "import torch"          (lingbotvla env)
#   ENV_NAME=RoboTwin scripts/hpc/run_in_env.sh python ...      (simulator env)
# On cluster: `apptainer exec --nv --overlay $OVERLAY:ro $IMAGE` + /ext3/env.sh + conda activate.
# Works on login nodes (no --nv needed for CPU-only commands; --nv is harmless there) and in jobs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
[[ $# -gt 0 ]] || { echo "usage: run_in_env.sh <command> [args...]" >&2; exit 2; }

if lingbot_use_overlay; then
  NV=""; [[ -e /dev/nvidia0 || -n "${SLURM_JOB_GPUS:-}${CUDA_VISIBLE_DEVICES:-}" ]] && NV="--nv"
  # :ro so several jobs can share the overlay; writes go to scratch/HF_HOME etc. (bound from the host).
  exec apptainer exec ${NV} --overlay "${OVERLAY}:ro" --pwd "${PWD}" "${IMAGE}" \
    bash -c 'source /ext3/env.sh >/dev/null 2>&1; conda activate "$0" 2>/dev/null || source activate "$0"; exec "$@"' "${ENV_NAME}" "$@"
else
  echo "run_in_env.sh: overlay ${OVERLAY} or image ${IMAGE} missing" >&2; exit 1
fi
