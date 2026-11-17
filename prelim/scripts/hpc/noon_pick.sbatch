#!/usr/bin/env bash
# Runs scripts/hpc/noon_pick.sh at a set time (10/26 12:00 Beijing = 00:00 EDT), then cancels the PENDING elements of
# the previous eval array OLD_ARRAY (they would keep running its old phases and never the noon pick).
#   OLD_ARRAY=<old eval array id> scripts/hpc/submit.sh --cpu scripts/hpc/noon_pick.sbatch --begin=2026-10-26T00:00:00
#SBATCH --job-name=lb_noonpick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/lingbot/logs/noonpick_%j.log
set -uo pipefail
cd "${WORK_DIR:-/scratch/${USER}/lingbot/LingBotRoboTwin}"
bash scripts/hpc/noon_pick.sh || { echo "pick failed: back to the queue $(date)"; scontrol requeue "${SLURM_JOB_ID}"; exit 0; }
if [[ -n "${OLD_ARRAY:-}" ]]; then
  P=$(timeout 120 squeue -h -u "${USER}" -j "${OLD_ARRAY}" -t PD -o %i | tr '\n' ' ')
  [[ -n "${P// /}" ]] && { echo "cancel pending ${P}"; scancel ${P}; }
fi
