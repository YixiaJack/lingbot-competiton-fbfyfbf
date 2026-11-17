#!/usr/bin/env bash
# Build the RoboTwin simulator env as a plain conda env on scratch ($PROJ/miniforge3/envs/RoboTwin), using the
# devel image only for its toolchain (nvcc 12.6 for curobo / pytorch3d). Installing into the overlay instead
# goes through fuse2fs and took >1.5 h for `pip install -r requirements.txt` alone.
# Steps = upstream RoboTwin script/_install.sh (pinned commit 13c3c47), inlined.
#   sbatch ... --wrap "bash scripts/hpc/build_robotwin_env.sh"   (CPU job is fine)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
MF="${PROJ}/miniforge3"
echo "=== RoboTwin env build on $(hostname) $(date) -> ${MF}/envs/${SIM_ENV}"
apptainer exec "${BUILD_IMAGE}" bash -s -- "${MF}" "${SIM_ENV}" "${ROBOTWIN_DIR}" "${PIP_CACHE_DIR}" "${SCRATCH}" <<'INNER'
set -euo pipefail
MF="$1"; SIM_ENV="$2"; ROBOTWIN_DIR="$3"; export PIP_CACHE_DIR="$4"; SCRATCH="$5"
export PYTHONNOUSERSITE=1 PIP_NO_INPUT=1
BUILD_TMP="${SCRATCH}/cache/build_tmp"; mkdir -p "${BUILD_TMP}"; cd "${BUILD_TMP}"; export TMPDIR="${BUILD_TMP}"
export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0" FORCE_CUDA=1 MAX_JOBS=16 CUDA_HOME=/usr/local/cuda
if [[ ! -x "${MF}/bin/conda" ]]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O mf.sh
  bash mf.sh -b -p "${MF}" && rm -f mf.sh
fi
source "${MF}/etc/profile.d/conda.sh"
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
python script/update_embodiment_config_path.py
python -c "import sapien, mplib, curobo, pytorch3d, torch; print('RoboTwin OK torch', torch.__version__, 'sapien', sapien.__version__)"
INNER
echo "=== done $(date)"
