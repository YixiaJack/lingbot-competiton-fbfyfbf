"""Render a grid of whitelist-augmented clean frames for visual inspection.

Each row is one task frame: original | 5 independent augmentation samples (head camera on top, wrist cameras below).
Fourier (p_fourier) and wrist dropout (p_wrist_drop, shown as a dark gray view) are drawn as in training.
  python src/preview_aug.py --raw_root $DATA_DIR/raw --out preview.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lingbot_aug  # noqa: E402

CAMS = ["head_camera", "left_camera", "right_camera"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=["lift_pot", "click_bell", "hanging_mug", "stack_blocks_three"])
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    rows = []
    for task in args.tasks:
        with h5py.File(args.raw_root / task / "aloha-agilex_clean_50" / "data" / "episode0.hdf5") as f:
            views = [cv2.imdecode(np.frombuffer(f[f"observation/{c}/rgb"][40], np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
                     for c in CAMS]
        imgs = torch.stack([torch.from_numpy(v.copy()).permute(2, 0, 1) for v in views])  # (3 views, C, H, W) uint8
        cols = [imgs]
        for _ in range(args.samples):
            params = lingbot_aug.sample_visual_augmentation_params(imgs[0])
            views = []
            for cam, v in zip(("top", "wrist_left", "wrist_right"), imgs):
                v = lingbot_aug.apply_visual_augmentation(v, params)
                if params.get("fourier"):
                    v = lingbot_aug.apply_fourier(v, params["fourier"], cam)
                if params.get("wrist_drop") and any(f"wrist_{s}" == cam for s in params["wrist_drop"]):
                    v = torch.full_like(v, 40)  # dropped wrist view (the model sees a masked camera)
                views.append(v)
            cols.append(torch.stack(views))
        # each column: head camera on top, the two wrist cameras side by side (half size) below
        tiles = []
        for col in cols:
            head = col[0].permute(1, 2, 0).numpy()
            wrists = np.concatenate([cv2.resize(col[i].permute(1, 2, 0).numpy(), (160, 120)) for i in (1, 2)], 1)
            tiles.append(np.concatenate([head, wrists], 0))
        rows.append(np.concatenate(tiles, 1))
    cv2.imwrite(args.out, np.concatenate(rows, 0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
