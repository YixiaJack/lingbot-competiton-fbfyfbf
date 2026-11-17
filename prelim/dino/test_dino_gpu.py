"""GPU checks for the DINOv2 fusion through the real inference loader (deploy.lingbot_vla_v2_policy).

  1. base hf_ckpt vs the zero-initialized DINO model_path: identical actions (same inputs, same seed)
  2. random projection -> actions change (the branch is live)
  3. optional: a trained DINO checkpoint loads strictly and runs
Each model dir must sit at <run>/checkpoints/<step>/hf_ckpt with <run>/lingbotvla_cli.yaml (deploy convention).

  PYTHONPATH=<dev checkout> python dino/test_dino_gpu.py --base <hf_ckpt> --dino_init <hf_ckpt> [--trained <hf_ckpt>]
"""

import argparse
import sys

import torch

from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def load(path):
    server = LingbotVLAv2Server(path_to_pi_model=path, use_bf16=False, use_fp32=True, use_length=50)
    return server.vla


def inputs(cfg, device):
    g = torch.Generator().manual_seed(1234)
    images = (torch.rand(1, 3, 256, 1536, generator=g) * 2 - 1).to(device)
    masks = torch.ones(1, 3, dtype=torch.bool, device=device)
    tokens = torch.randint(0, 1000, (1, 72), generator=g).to(device)
    lang_masks = torch.ones(1, 72, dtype=torch.bool, device=device)
    state = (torch.rand(1, cfg.max_state_dim, generator=g) * 2 - 1).to(device)
    thw = torch.tensor([[[1, 16, 16]] * 3], device=device)
    return images, masks, tokens, lang_masks, state, thw


@torch.no_grad()
def act(vla, x):
    images, masks, tokens, lang_masks, state, thw = x
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    return vla.model.sample_actions(images, masks, tokens, lang_masks, state, image_grid_thw=thw).float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dino_init", required=True)
    ap.add_argument("--trained", default=None)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    base = load(args.base)
    x = inputs(base.config, "cuda")
    a_base = act(base, x)
    a_base2 = act(base, x)
    noise = (a_base - a_base2).abs().max().item()  # some GPU kernels (MoE scatter) may not be bit-exact
    print(f"base run-to-run max abs diff {noise:.2e}", flush=True)
    del base
    torch.cuda.empty_cache()

    dino = load(args.dino_init)
    check("DINO model has the fusion branch", getattr(dino.model, "dino_fusion", None) is not None)
    check("loaded DINO weights are the pretrained ones (not zeros)",
          dino.model.dino_fusion.dino.embeddings.patch_embeddings.projection.weight.abs().sum().item() > 0)
    a_dino = act(dino, x)
    diff = (a_base - a_dino).abs().max().item()
    check("zero-init DINO model == base model (actions)", diff <= max(noise, 0.0) + 1e-6,
          f"max abs diff {diff:.2e} (base run-to-run {noise:.2e})")
    with torch.no_grad():
        dino.model.dino_fusion.proj.weight.normal_(0, 0.02)
    a_live = act(dino, x)
    check("non-zero projection changes the actions", (a_live - a_base).abs().max().item() > 100 * noise + 1e-4,
          f"max abs diff {(a_live - a_base).abs().max():.2e}")
    del dino
    torch.cuda.empty_cache()

    if args.trained:
        tr = load(args.trained)
        a_tr = act(tr, x)
        w = tr.model.dino_fusion.proj.weight
        check("trained checkpoint loads strictly and runs", torch.isfinite(a_tr).all().item(), tuple(a_tr.shape))
        check("trained projection moved away from zero", w.abs().max().item() > 0, f"|W|max {w.abs().max():.2e}")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
