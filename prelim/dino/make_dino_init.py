"""Build a model_path for DINOv2-fusion training from an existing LingBot hf_ckpt, without touching it.

LingBot's post-training loader requires every model parameter to be in model_path, so the new module's weights are
added as an extra shard: model.dino_fusion.dino.* = the public DINOv2 weights (frozen during training),
model.dino_fusion.norm = identity LayerNorm (ones / zeros), model.dino_fusion.proj = zeros (=> exact base model at
step 0). The base shards are symlinked and the index is merged.

  python dino/make_dino_init.py --base <run>/checkpoints/global_step_N/hf_ckpt \
      --dino $SCRATCH/lingbot/ckpts/dinov2-with-registers-base --out <dir> --out_dim 2560
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

PREFIX = "model.dino_fusion"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--dino", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--out_dim", type=int, default=2560)
    args = ap.parse_args()
    assert not args.out.exists(), f"{args.out} exists"
    args.out.mkdir(parents=True)
    index = json.loads((args.base / "model.safetensors.index.json").read_text())
    assert not any(k.startswith(PREFIX) for k in index["weight_map"]), "base already has dino_fusion weights"
    for f in args.base.iterdir():
        if f.name == "model.safetensors.index.json":
            continue
        if f.suffix == ".safetensors":
            os.symlink(f.resolve(), args.out / f.name)
        elif f.is_file():
            shutil.copy2(f, args.out / f.name)

    dino = load_file(str(args.dino / "model.safetensors"))
    hidden = dino["embeddings.patch_embeddings.projection.weight"].shape[0]
    extra = {f"{PREFIX}.dino.{k}": v.float().contiguous() for k, v in dino.items()}
    extra[f"{PREFIX}.norm.weight"] = torch.ones(hidden)
    extra[f"{PREFIX}.norm.bias"] = torch.zeros(hidden)
    extra[f"{PREFIX}.proj.weight"] = torch.zeros(args.out_dim, hidden)
    extra[f"{PREFIX}.proj.bias"] = torch.zeros(args.out_dim)
    shard = "model-dino-fusion.safetensors"
    save_file(extra, str(args.out / shard), metadata={"format": "pt"})

    index["weight_map"].update({k: shard for k in extra})
    index.setdefault("metadata", {})["total_size"] = index.get("metadata", {}).get("total_size", 0) + sum(
        v.numel() * v.element_size() for v in extra.values())
    (args.out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    n_frozen = sum(v.numel() for k, v in extra.items() if ".dino." in k)
    n_train = sum(v.numel() for k, v in extra.items() if ".dino." not in k)
    print(f"wrote {args.out}: {len(extra)} new tensors, frozen {n_frozen / 1e6:.2f}M, trainable {n_train / 1e6:.2f}M")


if __name__ == "__main__":
    main()
