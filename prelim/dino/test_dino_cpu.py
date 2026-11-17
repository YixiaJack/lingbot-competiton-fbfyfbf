"""CPU self-checks for the DINOv2 fusion (run in the lingbotvla env with the patched dev checkout first on PYTHONPATH).

  PYTHONPATH=<dev checkout>:<repo>/dino python dino/test_dino_cpu.py --qwen <Qwen3-VL-4B dir> \
      --dino <dinov2 dir> --frame_h5 <raw episode hdf5>
"""

import argparse
import sys

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoImageProcessor

import dino_fusion as D

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", required=True)
    ap.add_argument("--dino", required=True)
    ap.add_argument("--frame_h5", required=True)
    ap.add_argument("--img_size", type=int, default=256)
    args = ap.parse_args()

    # three real camera frames, resized like base_dataset (plain resize to img_size x img_size), uint8
    with h5py.File(args.frame_h5) as f:
        frames = [cv2.imdecode(np.frombuffer(f[f"observation/{c}/rgb"][40], np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
                  for c in ("head_camera", "left_camera", "right_camera")]
    imgs = torch.stack([torch.from_numpy(cv2.resize(fr, (args.img_size, args.img_size),
                                                    interpolation=cv2.INTER_LINEAR).copy()).permute(2, 0, 1)
                        for fr in frames])  # (3, 3, 256, 256) uint8

    # 1. exact inverse of the Qwen processor patchification
    proc = AutoImageProcessor.from_pretrained(args.qwen)
    pv = torch.stack([proc(images=im, return_tensors="pt")["pixel_values"] for im in imgs])  # (3, 256, 1536)
    grid = proc(images=imgs[0], return_tensors="pt")["image_grid_thw"]
    check("processor keeps 256x256 (grid 1x16x16)", grid.tolist() == [[1, 16, 16]], grid.tolist())
    rec = D.pixel_values_to_images(pv)
    err = (rec - imgs.float() / 255).abs().max().item()
    check("inverse reconstructs the image", err < 1.5 / 255, f"max abs err {err * 255:.3f} / 255")
    rec_bf16 = D.pixel_values_to_images(pv.to(torch.bfloat16))
    err16 = (rec_bf16 - imgs.float() / 255).abs().max().item()
    check("inverse after bf16 cast (training dtype)", err16 < 3 / 255, f"max abs err {err16 * 255:.3f} / 255")

    # 2. DINOv2 features of the reconstructed image vs the original image
    hidden = AutoConfig.from_pretrained(args.qwen).text_config.hidden_size
    check("LLM width from config", hidden == 2560, hidden)
    m = D.DinoFusion(args.dino, hidden)
    m.dino.load_state_dict(load_file(f"{args.dino}/model.safetensors"), strict=True)
    m.eval()
    with torch.no_grad():
        t_rec = m.dino_tokens(pv)
        x = F.interpolate(imgs.float() / 255, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        x = (x - torch.tensor(D.IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(D.IMAGENET_STD).view(1, 3, 1, 1)
        h = m.dino(pixel_values=x).last_hidden_state[:, m.num_prefix:]
        h = F.adaptive_avg_pool2d(h.transpose(1, 2).reshape(3, -1, 16, 16), 8).flatten(2).transpose(1, 2)
    cos = F.cosine_similarity(t_rec, h, dim=-1).min().item()
    check("DINO tokens from pixel_values == from image", cos > 0.999, f"min cosine {cos:.5f}")
    check("token grid", tuple(t_rec.shape) == (3, 64, 768), tuple(t_rec.shape))
    # spatial sanity: tokens differ across the image (not a constant), cameras differ from each other
    check("features vary over space", t_rec.std(dim=1).mean().item() > 0.05)

    # 3. zero init -> exact no-op; non-zero proj -> contribution
    with torch.no_grad():
        out = m(pv)
    check("zero-init output is exactly zero", out.abs().max().item() == 0.0 and tuple(out.shape) == (3, 64, hidden),
          tuple(out.shape))
    with torch.no_grad():
        m.proj.weight.normal_(0, 0.02)
        out2 = m(pv)
    check("non-zero proj gives a contribution", out2.abs().mean().item() > 0)
    m.train()
    check("train() keeps DINO in eval mode", not m.dino.training and m.proj.training)
    check("DINO frozen, proj/norm trainable", all(not p.requires_grad for p in m.dino.parameters())
          and m.proj.weight.requires_grad and m.norm.weight.requires_grad)

    # 4. the patched model module imports and contains the hooks
    import lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 as MV2
    src = open(MV2.__file__).read()
    check("patched checkout is the one imported", "dino_fusion" in src and "/dev/" in MV2.__file__, MV2.__file__)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
