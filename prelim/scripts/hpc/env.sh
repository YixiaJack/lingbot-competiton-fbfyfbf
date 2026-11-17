# Shared settings for every HPC script in this repo. Source it; do not run it.
#
# Adapted from Mamba2PD/scripts/hpc/env.sh. On cluster both python environments (lingbotvla: training +
# inference server; RoboTwin: simulator) live INSIDE one Apptainer overlay, so commands go through
# run_in_env.sh (apptainer exec --nv --overlay ... IMAGE). Per-user settings: ~/.config/lingbot/hpc.env.

export SCRATCH="${SCRATCH:-/scratch/${USER}}"
_cfg="${LINGBOT_HPC_ENV:-$HOME/.config/lingbot/hpc.env}"
if [[ -r "${_cfg}" ]]; then
  set -a; source "${_cfg}"; set +a
fi
unset _cfg

export SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"  # your allocation (e.g. set in ~/.config/lingbot/hpc.env)
export GPU_CONSTRAINT="${GPU_CONSTRAINT:-}"
# Runtime image: Ubuntu 24.04 (glibc 2.39). The host NVIDIA GL/Vulkan libs that --nv binds in need it (SAPIEN
# renders here, "Render Error" on 22.04), and its FFmpeg 6 libs let torchcodec decode the training videos.
export IMAGE="${IMAGE:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}"
# Build image: nvcc 12.6 toolchain for curobo / pytorch3d (build_overlay.sh, build_robotwin_env.sh only).
export BUILD_IMAGE="${BUILD_IMAGE:-/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}"
export OVERLAY="${OVERLAY:-${SCRATCH}/overlays/lingbot.ext3}"
export OVERLAY_TEMPLATE="${OVERLAY_TEMPLATE:-/share/apps/overlay-fs-ext3/overlay-50G-10M.ext3.gz}"
export ENV_NAME="${ENV_NAME:-lingbotvla}"
export SIM_ENV="${SIM_ENV:-RoboTwin}"

export PROJ="${PROJ:-${SCRATCH}/lingbot}"                 # everything for this competition
export LINGBOT_DIR="${LINGBOT_DIR:-${PROJ}/lingbot-vla-v2}"  # upstream training/inference code
export ROBOTWIN_DIR="${ROBOTWIN_DIR:-${PROJ}/RoboTwin}"      # RoboTwin @ 13c3c47 (pinned by LingBot)
export WORK_DIR="${WORK_DIR:-${PROJ}/LingBotRoboTwin}"       # this repo (rsynced from the laptop)
export CKPT_DIR="${CKPT_DIR:-${PROJ}/ckpts}"
export DATA_DIR="${DATA_DIR:-${PROJ}/data}"
export RUN_DIR="${RUN_DIR:-${PROJ}/runs}"
export LOG_DIR="${LOG_DIR:-${PROJ}/logs}"
export QWEN3VL_PATH="${QWEN3VL_PATH:-${CKPT_DIR}/Qwen3-VL-4B-Instruct}"

# caches off the 30k-inode home
export HF_HOME="${HF_HOME:-${SCRATCH}/.huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${DATA_DIR}/lerobot}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH}/cache/pip}"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH}/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH}/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${SCRATCH}/cache/inductor}"
export WANDB_DIR="${WANDB_DIR:-${SCRATCH}/wandb}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH}/apptainer-cache}"
export TMPDIR="${TMPDIR:-${SLURM_TMPDIR:-${SCRATCH}/TMPDIR}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONNOUSERSITE=1
mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${WANDB_DIR}" \
  "${APPTAINER_CACHEDIR}" "${TMPDIR}" "${CKPT_DIR}" "${DATA_DIR}" "${RUN_DIR}" "${LOG_DIR}" 2>/dev/null || true

[[ -n "${SLURM_ACCOUNT}" ]] && export SBATCH_ACCOUNT="${SLURM_ACCOUNT}" SLURM_ACCOUNT

lingbot_use_overlay() { [[ -n "${OVERLAY}" && -f "${OVERLAY}" && -f "${IMAGE}" ]] && command -v apptainer >/dev/null 2>&1; }
