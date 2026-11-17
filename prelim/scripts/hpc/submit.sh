#!/usr/bin/env bash
# Submit an sbatch script with the project's account and GPU constraint applied.
#   scripts/hpc/submit.sh scripts/hpc/smoke_train.sbatch            # GPU job: adds --account and --constraint
#   scripts/hpc/submit.sh --cpu scripts/hpc/download.sbatch   # CPU job: --account only
# Extra sbatch flags may follow the script path, e.g. ... smoke_train.sbatch --time=01:00:00
# Environment variables are exported to the job (--export=ALL).
# PREEMPT=1 (default) uses the cluster's preemption partitions: starts in minutes instead of days while the cluster is
# full, but the job may be cancelled after 30 min and is then requeued (every script here resumes / is idempotent).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
CPU=0; [[ "${1:-}" == "--cpu" ]] && { CPU=1; shift; }
SCRIPT="${1:?sbatch script required}"; shift
[[ -n "${SLURM_ACCOUNT}" ]] || { echo "SLURM_ACCOUNT is not set; set it in ~/.config/lingbot/hpc.env" >&2; exit 2; }
ARGS=(--account="${SLURM_ACCOUNT}" --export=ALL)
if [[ "${PREEMPT:-1}" == "1" ]]; then ARGS+=(--comment="preemption=yes;preemption_partitions_only=yes;requeue=true" --requeue); fi
if [[ "${CPU}" -eq 0 && -n "${GPU_CONSTRAINT}" ]]; then ARGS+=(--constraint="${GPU_CONSTRAINT}"); fi
mkdir -p "${LOG_DIR}"
echo "sbatch ${ARGS[*]} $* ${SCRIPT}"
sbatch "${ARGS[@]}" "$@" "${SCRIPT}"
