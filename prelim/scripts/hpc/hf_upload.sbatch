#!/usr/bin/env bash
# Upload a staged folder to a PRIVATE Hugging Face model repo from the cluster (resumable: upload_large_folder
# skips files already committed, so a preempted + requeued job just continues). Token: $HF_HOME/token.
#   SRC=/scratch/$USER/lingbot/hf_upload/main40k REPO=<hf-user>/lingbot-robotwin-main40k \
#     scripts/hpc/submit.sh --cpu scripts/hpc/hf_upload.sbatch
#SBATCH --job-name=lb_hfup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/lingbot/logs/hfup_%x_%j.log
set -euo pipefail
cd "${WORK_DIR:-/scratch/${USER}/lingbot/LingBotRoboTwin}"
source scripts/hpc/env.sh
: "${SRC:?}" "${REPO:?}"
echo "=== upload ${SRC} -> ${REPO} (private) $(date)"
"${PROJ}/.convvenv/bin/python" - <<EOF
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("${REPO}", repo_type="model", private=True, exist_ok=True)
assert api.repo_info("${REPO}").private, "repo is not private"
api.upload_large_folder(repo_id="${REPO}", folder_path="${SRC}", repo_type="model", private=True, num_workers=8)
EOF
echo "=== upload done $(date)"
