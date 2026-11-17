"""Init checkpoint for one arm of the routing experiment, without touching the base checkpoint.

LingBot's post-training loader wants every model parameter in model_path, so the new weights go into an extra shard:
  ...qwen_expert.model.layers.<i>.mlp.task_gates.<t>.weight-less name `task_gates.<t>` = copy of that layer's gate.weight
  ...qwenvl.model.language_model.layers.<i>.self_attn.kv_lora.{k,v}_A.<s> = kaiming-uniform (seed fixed per layer,
      identical for every set and every arm that uses that layer), {k,v}_B.<s> = zeros
so every arm equals the base model at step 0. Base shards are symlinked, the index is merged, route_config.json copied.

  python route/make_route_init.py --base ckpts/lingbot-vla-v2-6b --route route_config.json --out <dir>
"""
import argparse
import json
import math
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXP = "model.qwenvl_with_expert.qwen_expert.model.layers.{i}.mlp"
VLM = "model.qwenvl_with_expert.qwenvl.model.language_model.layers.{i}.self_attn"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--route", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cfg = json.loads(a.route.read_text())
    assert not a.out.exists(), f"{a.out} exists"
    index = json.loads((a.base / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    assert not any(".task_gates." in k or ".kv_lora." in k for k in wm), "base already has routing weights"

    def get(key):
        with safe_open(str(a.base / wm[key]), "pt") as f:
            return f.get_tensor(key)

    extra = {}
    n_routers = int(cfg.get("task_routers", 0))
    layers = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in wm if k.endswith(".mlp.gate.weight") and "qwen_expert" in k})
    for i in layers:
        g = get(f"{EXP.format(i=i)}.gate.weight")
        for t in range(n_routers):
            extra[f"{EXP.format(i=i)}.task_gates.{t}"] = g.clone().contiguous()
    lora = cfg.get("lora")
    if lora:
        r, sets = int(lora["rank"]), int(lora.get("sets", 1))
        for i in lora["layers"]:
            pre = VLM.format(i=i)
            kw, vw = get(f"{pre}.k_proj.weight"), get(f"{pre}.v_proj.weight")
            for name, w in (("k", kw), ("v", vw)):
                gen = torch.Generator().manual_seed(1000 + 2 * i + (name == "v"))
                bound = 1 / math.sqrt(w.shape[1])        # kaiming_uniform_(a=sqrt(5)) on (r, d_in)
                A = (torch.rand(r, w.shape[1], generator=gen) * 2 - 1) * bound
                for s in range(sets):
                    extra[f"{pre}.kv_lora.{name}_A.{s}"] = A.to(w.dtype).clone()   # own storage per set (safetensors refuses shared)
                    extra[f"{pre}.kv_lora.{name}_B.{s}"] = torch.zeros(w.shape[0], r, dtype=w.dtype)

    a.out.mkdir(parents=True)
    for f in a.base.iterdir():
        if f.name == "model.safetensors.index.json":
            continue
        if f.suffix == ".safetensors":
            os.symlink(f.resolve(), a.out / f.name)
        elif f.is_file():
            shutil.copy2(f, a.out / f.name)
        elif f.is_dir():
            os.symlink(f.resolve(), a.out / f.name)   # depth/, dino_video/ teachers
    shard = "model-route.safetensors"
    if extra:
        save_file(extra, str(a.out / shard), metadata={"format": "pt"})
        wm.update({k: shard for k in extra})
        index.setdefault("metadata", {})["total_size"] = index.get("metadata", {}).get("total_size", 0) + sum(
            v.numel() * v.element_size() for v in extra.values())
    (a.out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    shutil.copy2(a.route, a.out / "route_config.json")
    n_gate = sum(v.numel() for k, v in extra.items() if ".task_gates." in k)
    n_lora = sum(v.numel() for k, v in extra.items() if ".kv_lora." in k)
    print(json.dumps({"out": str(a.out), "tensors": len(extra), "task_gate_params": n_gate, "lora_params": n_lora}))


if __name__ == "__main__":
    main()
