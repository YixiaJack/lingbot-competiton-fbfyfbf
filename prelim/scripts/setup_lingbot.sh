#!/usr/bin/env bash
# Set up the lingbot-vla-v2 checkout used for training / inference: upstream at the pinned commit + this repo's patches.
#   VARIANT=base  (default) upstream + src/patch_upstream.py (env-gated augmentation / sampling; inert when unset)
#   VARIANT=dino  additionally dino/patch_dino.py (DINOv2 fusion, active only when train.dino_fusion_path is set)
#   DEST=/path/to/lingbot-vla-v2[-dino] VARIANT=dino bash scripts/setup_lingbot.sh
# Training / serving the 40k model needs no patch at all; the patched code behaves identically for it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${VARIANT:-base}"; COMMIT=be969b8fd117fb70550c5d4bf4bc328211b5b1b6
DEST="${DEST:-${PWD}/lingbot-vla-v2$([[ ${VARIANT} == dino ]] && echo -dino)}"
[[ ${VARIANT} == dino && "$(basename "${DEST}")" != *dino* ]] && { echo "DEST name must contain 'dino' (patch_dino.py guard)" >&2; exit 1; }
[[ -d "${DEST}/.git" ]] || git clone https://github.com/robbyant/lingbot-vla-v2.git "${DEST}"
git -C "${DEST}" checkout -q "${COMMIT}"
python3 "${HERE}/src/patch_upstream.py" "${DEST}"
[[ ${VARIANT} == dino ]] && python3 "${HERE}/dino/patch_dino.py" "${DEST}"
git -C "${DEST}" diff --stat
echo "ready: LINGBOT_DIR=${DEST}  (put ${HERE}/src on PYTHONPATH when training with the augmentation env vars)"
