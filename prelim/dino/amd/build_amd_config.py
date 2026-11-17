"""Write the AMD training config for the DINOv2-fusion continuation from the config the cluster run actually used.

Input: configs/dino_cont8k_6k_lingbotvla_cli.yaml (written by the cluster training program, cluster paths). Output: the same
recipe with AMD paths and the few settings the AMD hosts need:
  * paths: data list, clean norm stats, robot configs, init checkpoint, Qwen tokenizer, depth / video teachers, DINOv2
  * memory (48 GB W7900D instead of 80-180 GB): full FSDP sharding over all GPUs + gradient checkpointing
  * torch.compile off (the AMD hosts' own runs keep it off on ROCm)
  * arguments the AMD code base (upstream 951475a) does not have are dropped: dist_muon_* , precompute_grid_thw
Everything else (lr 1e-5 -> 5e-6, warmup 3%, global batch 128, Muon, image_augment, future image, align losses, DINO
fusion grid / input size, 6000 steps) is copied unchanged.

  python build_amd_config.py --cluster <cluster yaml> --out <amd yaml> --pkg <package dir> --ngpu 8 --micro 16 ...
"""

import argparse
import os

import yaml

DROP = ("dist_muon_attn_per_head", "dist_muon_layers_per_bucket", "precompute_grid_thw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pkg", required=True, help="package dir (configs/, data list, norm stats)")
    ap.add_argument("--init", required=True, help="DINO init hf_ckpt (make_dino_init.py output)")
    ap.add_argument("--dinov2", default=None, help="facebook/dinov2-with-registers-base dir (not needed with --no_dino)")
    ap.add_argument("--models", required=True, help="AMD models dir (Qwen tokenizer, lingbot-vla-v2-6b, moge)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ngpu", type=int, default=8)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=None, help="override the cluster config's max_steps (cosine schedule length)")
    ap.add_argument("--lr", type=float, default=None, help="override train.lr")
    ap.add_argument("--lr_min", type=float, default=None, help="override train.lr_min")
    ap.add_argument("--no_dino", action="store_true", help="plain continuation: no DINOv2 fusion, --init is the base hf_ckpt")
    args = ap.parse_args()

    c = yaml.safe_load(open(args.cluster))
    d, m, t = c["data"], c["model"], c["train"]
    pkg, models = os.path.abspath(args.pkg), os.path.abspath(args.models)
    base = f"{models}/robbyant_lingbot-vla-v2-6b"

    d["train_path"] = f"{pkg}/configs/robotwin_clean_amd.txt"
    d["norm_stats_file"] = f"{pkg}/configs/norm_stats/robotwin_clean.json"
    d["robot_config_root"] = f"{pkg}/configs/robot_configs_amd"
    m["model_path"] = m["config_path"] = os.path.abspath(args.init)
    m["tokenizer_path"] = f"{models}/Qwen3-VL-4B-Instruct-config-tokenizer"
    ap_ = t["align_params"]
    ap_["depth"]["moge_path"] = f"{models}/moge-2-vitb-normal/model.pt"
    ap_["depth"]["morgbd_path"] = f"{base}/depth/model.pt"
    ap_["video"]["ckpt_path"] = f"{base}/dino_video/teacher_step_10000.pth"
    ap_["video"]["config_path"] = f"{base}/dino_video/config.yaml"
    if args.no_dino:
        for k in ("dino_fusion_path", "dino_fusion_input_size", "dino_fusion_grid"):
            t.pop(k, None)
    else:
        assert args.dinov2, "--dinov2 required unless --no_dino"
        t["dino_fusion_path"] = os.path.abspath(args.dinov2)
    t["output_dir"] = os.path.abspath(args.output_dir)

    assert t["global_batch_size"] % (args.ngpu * args.micro) == 0, "global batch must be a multiple of ngpu * micro"
    t["micro_batch_size"] = args.micro
    t["gradient_accumulation_steps"] = t["global_batch_size"] // (args.ngpu * args.micro)
    t["data_parallel_shard_size"] = args.ngpu
    t["data_parallel_replicate_size"] = 1
    t["enable_full_shard"] = True
    t["enable_gradient_checkpointing"] = True
    t["use_compile"] = False
    t["save_steps"] = args.save_steps
    if args.max_steps:
        t["max_steps"] = args.max_steps
    if args.lr is not None:
        t["lr"] = args.lr
    if args.lr_min is not None:
        t["lr_min"] = args.lr_min
    t["enable_resume"] = True
    for k in DROP:
        t.pop(k, None)

    for p in (d["norm_stats_file"], m["tokenizer_path"], ap_["depth"]["moge_path"], ap_["depth"]["morgbd_path"],
              ap_["video"]["ckpt_path"], ap_["video"]["config_path"]) + (() if args.no_dino else (t["dino_fusion_path"],)):
        assert os.path.exists(p), f"missing {p}"
    with open(args.out, "w") as f:
        yaml.safe_dump(c, f, sort_keys=False, allow_unicode=True)
    print(f"wrote {args.out}: micro {args.micro} x {args.ngpu} GPUs x accum {t['gradient_accumulation_steps']} "
          f"= {t['global_batch_size']}, lr {t['lr']} -> {t['lr_min']}, {t['max_steps']} steps, optimizer {t['optimizer']}, "
          f"{'no DINO' if args.no_dino else 'DINO fusion'}, init {m['model_path']}")


if __name__ == "__main__":
    main()
