#!/usr/bin/env bash
# Start (or resume) the 22k+8k continuation on the AMD server in the background (no DINO). Run from the package root
# after dino/amd/setup_amd.sh:
#   bash dino/amd/train_amd.sh                      # all 8 GPUs, detached; log: runs/logs/cont_amd_<time>.log
#   MAX_STEPS=20 SAVE=10 bash dino/amd/train_amd.sh        # smoke test (writes to runs/cont_amd_smoke)
# Same recipe as the 10/23 AMD run: whitelist augmentation + Fourier + wrist dropout
# (AUG, default configs/aug_strong_v3.json: 90% of samples augmented, + random convolution, stronger Fourier / color,
# vertical shift for table height), seen-pool instruction swap, state noise 0.01, task sampling TASK_SAMPLING (default temp:2,
# as the cluster run from step 500 on; "uniform" = the original recipe), low-motion frame weighting, ViT lr x0.1.
# Resumes from the newest checkpoint in the output dir.
# Refuses to start while other processes hold GPU memory (e.g. an evaluation) unless FORCE=1.
set -euo pipefail
PKG="$(pwd)"
WORK="${WORK:-${PKG}}"
PY="${PY:-/opt/robotwin-env/bin/python}"
CODE="${WORK}/code/lingbot-vla-v2-dino"
NGPU="${NGPU:-8}"
GPU_IDS="${GPU_IDS:-$(seq -s, 0 $((NGPU - 1)))}"
CFG="${CFG:-${PKG}/configs/cont_amd_nodepth4k.yaml}"  # e.g. CFG=$PWD/configs/cont_amd_nodepth3k.yaml
[[ -f "${CFG}" ]] || { echo "missing ${CFG}: run dino/amd/setup_amd.sh first (from the package root)" >&2; exit 1; }  # CFG=<yaml> for e.g. a smoke test on fewer GPUs
EXTRA=()
if [[ -n "${MAX_STEPS:-}" ]]; then  # smoke test: separate output dir, short run
  OUT="${WORK}/runs/cont_amd_smoke"
  EXTRA+=(--train.max_steps "${MAX_STEPS}" --train.save_steps "${SAVE:-${MAX_STEPS}}" --train.output_dir "${OUT}")
else
  OUT=$("${PY}" -c "import yaml; print(yaml.safe_load(open('${CFG}'))['train']['output_dir'])")
fi
if [[ "${FORCE:-0}" != 1 ]]; then
  BUSY=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{gsub(/[^0-9]/,"",$NF); if ($NF > 2e9) n++} END{print n+0}')
  [[ "${BUSY}" -eq 0 ]] || { echo "${BUSY} GPU(s) already hold > 2 GB (evaluation running?); FORCE=1 to start anyway" >&2; exit 1; }
fi
mkdir -p "${OUT}" "${WORK}/runs/logs"
LOG="${WORK}/runs/logs/cont_amd_$(date +%Y%m%d_%H%M%S).log"

export HIP_VISIBLE_DEVICES="${GPU_IDS}"; unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONPATH="${CODE}:${PKG}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LINGBOT_AUG_CONFIG="${PKG}/${AUG:-configs/aug_strong_v3.json}" LINGBOT_SEEN_POOL=1 LINGBOT_STATE_NOISE=0.01 \
       LINGBOT_VISION_LR_SCALE=0.1 LINGBOT_TASK_SAMPLING="${TASK_SAMPLING:-temp:2}" LINGBOT_FRAME_WEIGHTS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
# the hosts' ROCm stack: without these the aiter gluon kernels (need triton>=3.6, server has 3.5.1) fail at import and the
# LingBot model classes never get registered (same settings as their launch_official_server.sh)
export AITER_TRITON_ONLY=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
# /tmp on the AMD box is a 4 GB tmpfs; the first step's GPU kernel compilation (comgr, ~0.5 GB per rank) fills it and
# the ranks die with "LLVM ERROR: ... No space left on device" + SIGSEGV. Keep all temp / compile caches on the big disk.
export TMPDIR="${WORK}/tmp" TRITON_CACHE_DIR="${WORK}/tmp/triton" TORCHINDUCTOR_CACHE_DIR="${WORK}/tmp/inductor"
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
cd "${CODE}"
{
  echo "=== cont_amd $(date) | GPUs ${GPU_IDS} | out ${OUT} | config ${CFG}"
  env | grep -E "^(LINGBOT_|PYTHONPATH=|HIP_VISIBLE_DEVICES=|AITER_|FLASH_ATTENTION_|TMPDIR=|TRITON_CACHE_DIR=)" | sed 's/^/    /'
} > "${LOG}"
nohup setsid "${PY}" -m torch.distributed.run --standalone --nproc-per-node="${NGPU}" \
  -m tasks.vla.train_lingbotvla "${CFG}" "${EXTRA[@]}" >> "${LOG}" 2>&1 < /dev/null &
echo $! > "${OUT}/train.pid"
echo "started pid $(cat "${OUT}/train.pid"), log ${LOG}"
echo "follow: tail -f ${LOG} | tr '\\r' '\\n' | grep -E 'Step|Traceback|Error'"
