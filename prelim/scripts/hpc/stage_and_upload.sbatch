#!/usr/bin/env bash
# Stage one checkpoint of a run in the submission layout and upload it to a PRIVATE Hugging Face model repo, then
# verify every file in the repo against local sizes and SHA256 (weights). Meant to be chained after training:
#   RUN_NAME=strong_main40k_10k STEP=10000 REPO=<hf-user>/lingbot-robotwin-strong-main40k-10k \
#     scripts/hpc/submit.sh --cpu scripts/hpc/stage_and_upload.sbatch --dependency=afterok:<train job>
# Layout: 03_模型权重/checkpoint/<RUN_NAME>/{lingbotvla_cli.yaml,model_assets/,checkpoints/global_step_<STEP>/hf_ckpt}
# (weights are hard links, no copy) + 02_代码材料/ (this repo) + README.md. Token: $HF_HOME/token.
#SBATCH --job-name=lb_stage_up
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/lingbot/logs/hfup_%x_%j.log
set -euo pipefail
cd "${WORK_DIR:-/scratch/${USER}/lingbot/LingBotRoboTwin}"
source scripts/hpc/env.sh
: "${RUN_NAME:?}" "${STEP:?}" "${REPO:?}"
RUN="${RUN_DIR}/${RUN_NAME}"
CK="${RUN}/checkpoints/global_step_${STEP}/hf_ckpt"
[[ -f "${CK}/model.safetensors.index.json" ]] || { echo "missing ${CK}"; exit 1; }
SRC="${PROJ}/hf_upload/${RUN_NAME}_${STEP}"
DST="${SRC}/03_模型权重/checkpoint/${RUN_NAME}"
echo "=== stage ${CK} -> ${SRC} $(date)"
mkdir -p "${DST}/checkpoints/global_step_${STEP}" "${SRC}/02_代码材料"
[[ -d "${DST}/checkpoints/global_step_${STEP}/hf_ckpt" ]] || cp -al "${CK}" "${DST}/checkpoints/global_step_${STEP}/"
cp -f "${RUN}/lingbotvla_cli.yaml" "${DST}/"
rsync -a "${RUN}/model_assets" "${DST}/"
rsync -a --delete --exclude .git --exclude __pycache__ --exclude docs \
  README.md train.sh eval.sh configs src scripts patches "${SRC}/02_代码材料/"
(cd "${DST}/checkpoints/global_step_${STEP}/hf_ckpt" && sha256sum *.safetensors *.json) > "${DST}/checkpoints/global_step_${STEP}/SHA256SUMS.txt"
cat > "${SRC}/README.md" <<EOF
# LingBot-VLA 2.0 RoboTwin clean-only: ${RUN_NAME} / global_step_${STEP}

- 权重：\`03_模型权重/checkpoint/${RUN_NAME}/checkpoints/global_step_${STEP}/hf_ckpt\`
  （推理服务从 \`hf_ckpt/../../../lingbotvla_cli.yaml\` 读取训练配置，请保持目录层级）
- 训练配置：\`03_模型权重/checkpoint/${RUN_NAME}/lingbotvla_cli.yaml\`（训练实际生效的完整配置）
- 代码：\`02_代码材料/\`（说明见其中的 README.md）
- 校验：\`SHA256SUMS.txt\`
- 训练只使用 RoboTwin 2.0 Aloha-AgileX 50 个任务的 clean 数据，没有使用 randomized 数据。
EOF
echo "=== upload -> ${REPO} (private) $(date)"
"${PROJ}/.convvenv/bin/python" - <<EOF
import hashlib, os
from huggingface_hub import HfApi
api, repo, src = HfApi(), "${REPO}", "${SRC}"
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
assert api.repo_info(repo).private, "repo is not private"
api.upload_large_folder(repo_id=repo, folder_path=src, repo_type="model", private=True, num_workers=8)
info = api.repo_info(repo, files_metadata=True)
remote = {s.rfilename: s for s in info.siblings}
bad = 0; n = 0
for d, _, fs in os.walk(src):
    if ".cache" in d:
        continue
    for f in fs:
        p = os.path.join(d, f); rel = os.path.relpath(p, src); s = remote.get(rel); n += 1
        if s is None or s.size != os.path.getsize(p):
            bad += 1; print("MISSING/SIZE", rel); continue
        if s.lfs:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            if h.hexdigest() != s.lfs.sha256:
                bad += 1; print("SHA MISMATCH", rel)
print(f"verified {n} files, private={info.private}, mismatches={bad}")
assert bad == 0
EOF
echo "=== upload verified $(date)"
"${PROJ}/.convvenv/bin/python" scripts/hpc/hf_readme_normstats.py "${REPO}"  # clean-only norm stats notice
