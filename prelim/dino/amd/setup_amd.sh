#!/usr/bin/env bash
# One-time setup of the 22k+8k continuation on the AMD server (8x W7900D, ROCm, no internet). Idempotent. No DINO:
# nothing here needs DINOv2 weights (the 10/23 AMD run was the same plain continuation).
# Run from the package root (the directory holding dino/, src/, configs/):
#   bash dino/amd/setup_amd.sh
# Needs on the server (paths overridable by environment variables):
#   AMD_LINGBOT_SRC  the hosts' lingbot-vla-v2 copy (upstream 951475a + their ROCm / LeRobot 0.6 edits), in the image
#   AMD_MODELS       Qwen3-VL tokenizer, robbyant_lingbot-vla-v2-6b (depth / dino_video teachers), moge-2-vitb-normal
#   BASE_CKPT        strong_scratch22k_cont8k / global_step_8000 hf_ckpt (the starting point)
#   DATA_DIR         our clean LeRobot dataset (50 tasks, with meta/seen_instructions.json and meta/frame_weights.npy)
# Produces under WORK (default: the package root):
#   code/lingbot-vla-v2-dino   hosts' code + src/patch_upstream.py + dino/patch_dino.py (code only; inactive without
#                              dino_fusion_path, kept so the code is the 10/23 run's) + dino/patch_amd.py
#   configs/robotwin_clean_amd.txt, configs/robot_configs_amd/, configs/cont_amd.yaml and the config that is trained:
#   configs/cont_amd_nodepth4k.yaml = cont_amd.yaml with depth / future-depth loss weights 1e-8 (the AMD depth teacher
#   gives losses ~4 vs ~0.5 at cluster, see the 10/23 run) and TOTAL_STEPS (default 4000) steps
set -euo pipefail
PKG="$(pwd)"; [[ -f "${PKG}/dino/patch_dino.py" ]] || { echo "run from the package root" >&2; exit 1; }
WORK="${WORK:-${PKG}}"
PY="${PY:-/opt/robotwin-env/bin/python}"
AMD_LINGBOT_SRC="${AMD_LINGBOT_SRC:-/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2}"
AMD_MODELS="${AMD_MODELS:-/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models}"
BASE_CKPT="${BASE_CKPT:-/root/robotwin-scratch/checkpoints/strong_scratch22k_cont8k/global_step_8000/hf_ckpt}"
DATA_DIR="${DATA_DIR:-${WORK}/data/robotwin_clean}"
NGPU="${NGPU:-8}"; MICRO="${MICRO:-16}"; SAVE="${SAVE:-500}"; TOTAL_STEPS="${TOTAL_STEPS:-4000}"
LR="${LR:-5e-6}"; LR_MIN="${LR_MIN:-2e-6}"
DEPTH_W="${DEPTH_W:-1.0e-8}"
OUT_DIR="${OUT_DIR:-${WORK}/runs/cont8k_v3_nodepth4k_amd}"
CFG_OUT="${CFG_OUT:-${PKG}/configs/cont_amd_nodepth4k.yaml}"
NORM_SHA=92b37257234b37c8642894997847da496c6e6bb550c2968eced4439c8b6bed31
CODE="${WORK}/code/lingbot-vla-v2-dino"

echo "== checks"
for p in "${AMD_LINGBOT_SRC}/tasks/vla/train_lingbotvla.py" "${BASE_CKPT}/model.safetensors.index.json" "${DATA_DIR}"; do
  [[ -e "${p}" ]] || { echo "missing ${p}" >&2; exit 1; }
done
for s in $(python3 -c "import json,sys; print(' '.join(sorted(set(json.load(open(sys.argv[1]))['weight_map'].values()))))" \
           "${BASE_CKPT}/model.safetensors.index.json"); do
  [[ -s "${BASE_CKPT}/${s}" ]] || { echo "missing shard ${BASE_CKPT}/${s}" >&2; exit 1; }
done
[[ "$(sha256sum configs/norm_stats/robotwin_clean.json | cut -c1-64)" == "${NORM_SHA}" ]] \
  || { echo "configs/norm_stats/robotwin_clean.json is not the clean-only stats" >&2; exit 1; }
NT=$(ls -d "${DATA_DIR}"/*/ | wc -l); [[ "${NT}" -eq 50 ]] || { echo "expected 50 datasets in ${DATA_DIR}, found ${NT}" >&2; exit 1; }
for d in "${DATA_DIR}"/*/; do
  for f in meta/info.json meta/seen_instructions.json meta/frame_weights.npy; do
    [[ -f "${d}${f}" ]] || { echo "missing ${d}${f}" >&2; exit 1; }
  done
done

echo "== code: ${CODE}"
if [[ ! -f "${CODE}/lingbotvla/models/vla/lingbot_vla/dino_fusion.py" ]]; then
  rm -rf "${CODE}"; mkdir -p "${CODE}"
  (cd "${AMD_LINGBOT_SRC}" && git ls-files -z | xargs -0 -I{} cp --parents {} "${CODE}/" && cp -r .git "${CODE}/")
fi
python3 src/patch_upstream.py "${CODE}" | tail -1
(cd dino && python3 patch_dino.py "${CODE}" | tail -1)
python3 dino/patch_amd.py "${CODE}"

echo "== data list / robot config"
mkdir -p configs/robot_configs_amd
for d in "${DATA_DIR}"/*/; do echo "robotwin_clean ${d%/}"; done > configs/robotwin_clean_amd.txt
sed "s#^norm_stats:.*#norm_stats: ${PKG}/configs/norm_stats/robotwin_clean.json#" \
  configs/robot_configs/robotwin_clean.yaml > configs/robot_configs_amd/robotwin_clean.yaml

echo "== config"
"${PY}" dino/amd/build_amd_config.py --cluster configs/dino_cont8k_6k_lingbotvla_cli.yaml --out configs/cont_amd.yaml \
  --no_dino --pkg "${PKG}" --init "${BASE_CKPT}" --models "${AMD_MODELS}" --output_dir "${OUT_DIR}" \
  --ngpu "${NGPU}" --micro "${MICRO}" --save_steps "${SAVE}" --max_steps "${TOTAL_STEPS}" --lr "${LR}" --lr_min "${LR_MIN}"
"${PY}" - "${PKG}/configs/cont_amd.yaml" "${CFG_OUT}" "${DEPTH_W}" <<'EOF'
import sys, yaml
src, dst, w = sys.argv[1], sys.argv[2], float(sys.argv[3])
c = yaml.safe_load(open(src))
a = c["train"]["align_params"]
a["depth_loss_weight"] = w
a["future_depth_loss_weight"] = w
yaml.safe_dump(c, open(dst, "w"), sort_keys=False, allow_unicode=True)
t = c["train"]
print(f"wrote {dst}")
for k in ("lr", "lr_min", "lr_warmup_ratio", "lr_decay_style", "max_steps", "save_steps", "global_batch_size",
          "micro_batch_size", "gradient_accumulation_steps", "optimizer", "enable_full_shard",
          "enable_gradient_checkpointing", "use_compile", "enable_resume", "output_dir"):
    print(f"    train.{k}: {t.get(k)}")
print(f"    depth_loss_weight: {a['depth_loss_weight']}  future_depth_loss_weight: {a['future_depth_loss_weight']}  "
      f"future_video_loss_weight: {a.get('future_video_loss_weight')}")
print(f"    model_path: {c['model']['model_path']}")
print(f"    norm_stats_file: {c['data']['norm_stats_file']}  train_path: {c['data']['train_path']}")
assert "dino_fusion_path" not in t
EOF
echo "setup done. Train: bash dino/amd/train_amd.sh   (config ${CFG_OUT})"
