#!/usr/bin/env bash
# Build the project's Apptainer overlay on cluster (pattern from Mamba2PD/scripts/hpc/torch_build_overlay.sh):
#   overlay template -> Miniforge in /ext3 -> two conda envs:
#     lingbotvla : python 3.12, torch 2.8.0, flash-attn 2.8.3, lerobot 0.4.2, depth packages
#                  (upstream tools/create_train_env.sh, unchanged)
#     RoboTwin   : python 3.10, torch 2.4.1, sapien 3.0.0b1, mplib 0.2.1, pytorch3d, curobo v0.7.8
#                  (upstream RoboTwin script/_install.sh steps, inlined so CUDA arch / temp dirs are set)
# Runs anywhere: on a login node (bash scripts/hpc/build_overlay.sh) or as a job (build_overlay.sbatch).
# Without a GPU, upstream's torch.cuda.is_available() assert is dropped from a copy of its script and the
# CUDA extensions are compiled for TORCH_CUDA_ARCH_LIST (A100 / L40S / H100+H200).
# Re-running is safe: every step checks what already exists.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
echo "=== overlay build on $(hostname) $(date)"
echo "  image    : ${BUILD_IMAGE}"
echo "  overlay  : ${OVERLAY}"
[[ -d "${LINGBOT_DIR}/.git" && -d "${ROBOTWIN_DIR}/.git" ]] || { echo "clone code first (download.sbatch)"; exit 1; }

mkdir -p "$(dirname "${OVERLAY}")"
if [[ ! -f "${OVERLAY}" ]]; then
  # sparse 60 GB ext3 (instant; gunzipping the 50G template writes ~8 MB/s on scratch)
  apptainer overlay create --sparse --size 61440 "${OVERLAY}"
fi

NV=""; [[ -e /dev/nvidia0 ]] && NV="--nv"
apptainer exec ${NV} --overlay "${OVERLAY}:rw" "${BUILD_IMAGE}" bash -s -- \
  "${ENV_NAME}" "${SIM_ENV}" "${LINGBOT_DIR}" "${ROBOTWIN_DIR}" "${PIP_CACHE_DIR}" "${SCRATCH}" <<'INNER'
set -euo pipefail
ENV_NAME="$1"; SIM_ENV="$2"; LINGBOT_DIR="$3"; ROBOTWIN_DIR="$4"; PIP_CACHE_DIR="$5"; SCRATCH="$6"
export PIP_CACHE_DIR PYTHONNOUSERSITE=1 PIP_NO_INPUT=1
unset -f which 2>/dev/null || true
# flash-attn's setup.py os.rename()s a downloaded wheel from CWD into pip's cache: keep both on $SCRATCH (EXDEV).
BUILD_TMP="${SCRATCH}/cache/build_tmp"; mkdir -p "${BUILD_TMP}"; cd "${BUILD_TMP}"; export TMPDIR="${BUILD_TMP}"
export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0" FORCE_CUDA=1 MAX_JOBS=16 CUDA_HOME=/usr/local/cuda

if [[ ! -x /ext3/miniforge3/bin/conda ]]; then
  echo "installing Miniforge into /ext3/miniforge3"
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O mf.sh
  bash mf.sh -b -p /ext3/miniforge3 && rm -f mf.sh
fi
cat > /ext3/env.sh <<'EOF'
#!/bin/bash
unset -f which
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
export PYTHONNOUSERSITE=1
EOF
source /ext3/env.sh
conda config --remove channels defaults >/dev/null 2>&1 || true

# ---------------- lingbotvla ----------------
if [[ ! -f /ext3/.lingbotvla_ok ]]; then
  echo "=== building ${ENV_NAME} $(date)"
  cd "${LINGBOT_DIR}"
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then RESUME=--resume; else RESUME=; fi
  ENV_SCRIPT=tools/create_train_env.sh
  if ! nvidia-smi >/dev/null 2>&1; then
    sed '/assert torch.cuda.is_available()/d' tools/create_train_env.sh > tools/.create_train_env_nogpu.sh
    ENV_SCRIPT=tools/.create_train_env_nogpu.sh
  fi
  bash "${ENV_SCRIPT}" --env-name "${ENV_NAME}" ${RESUME}
  conda activate "${ENV_NAME}"
  pip install -q tyro
  python -c "import torch, flash_attn, lerobot, lingbotvla; print('lingbotvla OK torch', torch.__version__, 'fa', flash_attn.__version__)"
  conda deactivate
  touch /ext3/.lingbotvla_ok
  cd "${BUILD_TMP}"
fi

# ---------------- RoboTwin ----------------
if [[ ! -f /ext3/.robotwin_ok ]]; then
  echo "=== building ${SIM_ENV} $(date)"
  conda env list | awk '{print $1}' | grep -qx "${SIM_ENV}" || conda create -y -n "${SIM_ENV}" python=3.10
  conda activate "${SIM_ENV}"
  cd "${ROBOTWIN_DIR}"
  pip install -q -r script/requirements.txt
  python -c "import pytorch3d" 2>/dev/null || pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
  SAPIEN_LOCATION=$(pip show sapien | grep 'Location' | awk '{print $2}')/sapien
  sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "${SAPIEN_LOCATION}/wrapper/urdf_loader.py"
  MPLIB_LOCATION=$(pip show mplib | grep 'Location' | awk '{print $2}')/mplib
  sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "${MPLIB_LOCATION}/planner.py"
  if ! python -c "import curobo" 2>/dev/null; then
    [[ -d envs/curobo ]] || git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git envs/curobo
    (cd envs/curobo && pip install -e . --no-build-isolation)
  fi
  pip install -q warp-lang==1.12.0 setuptools==69.5.1 msgpack websockets typing_extensions
  python -c "import sapien, mplib, curobo, pytorch3d, torch; print('RoboTwin OK torch', torch.__version__, 'sapien', sapien.__version__)"
  conda deactivate
  touch /ext3/.robotwin_ok
fi
echo "=== overlay done $(date)"; df -h /ext3 | tail -1
INNER
