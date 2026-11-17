"""Make a checkpoint servable on another machine: localize the paths in its training config and pin the norm stats.

The LingBot inference server reads `<hf_ckpt>/../../../lingbotvla_cli.yaml` (the config the training run wrote) and takes
from it the Qwen3-VL tokenizer path, the normalization statistics (data.norm_stats_file) and, for the DINO-fusion
model, train.dino_fusion_path. Those are absolute paths on our cluster. This script builds

  <stage>/<run>/lingbotvla_cli.yaml                      localized copy of the training config
  <stage>/<run>/checkpoints/<step>/hf_ckpt -> <hf_ckpt>  symlink (weights are not copied)

and prints the staged hf_ckpt path. Rewrites (plain text, so no PyYAML is needed):
  .../ckpts/<x>                -> $CKPT_DIR/<x>          (Qwen3-VL-4B-Instruct, lingbot-vla-v2-6b, moge, ...)
  .../LingBotRoboTwin/<x>      -> <this repo>/<x>
  model.config_path / model_path under .../runs/ (continuation init) -> the evaluated hf_ckpt
  data.norm_stats_file         -> <this repo>/configs/norm_stats/robotwin_clean.json, SHA-256 checked
  train.dino_fusion_path       -> <this repo>/configs/dinov2-with-registers-base (config.json only: the DINOv2
                                  weights are inside hf_ckpt, model.dino_fusion.dino.*)

  python3 src/stage_ckpt.py --ckpt <hf_ckpt> --ckpt_dir <dir with Qwen3-VL-4B-Instruct ...> --stage <dir>
"""

import argparse
import hashlib
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
NORM_STATS = HERE / "configs/norm_stats/robotwin_clean.json"
NORM_SHA256 = "92b37257234b37c8642894997847da496c6e6bb550c2968eced4439c8b6bed31"  # clean-only stats (README)
DINO_CONFIG_DIR = HERE / "configs/dinov2-with-registers-base"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="<run>/checkpoints/global_step_N/hf_ckpt")
    ap.add_argument("--ckpt_dir", type=Path, required=True)
    ap.add_argument("--stage", type=Path, required=True)
    args = ap.parse_args()
    ckpt = Path(os.path.abspath(args.ckpt))  # lexical, like the server's Path(model_path).parent.parent.parent
    run, step = ckpt.parent.parent.parent, ckpt.parent.name
    src_yaml = run / "lingbotvla_cli.yaml"
    assert (ckpt / "model.safetensors.index.json").is_file() or (ckpt / "model.safetensors").is_file(), f"no weights in {ckpt}"
    assert src_yaml.is_file(), f"missing {src_yaml} (keep the <run>/checkpoints/<step>/hf_ckpt layout)"

    sha = hashlib.sha256(NORM_STATS.read_bytes()).hexdigest()
    assert sha == NORM_SHA256, f"{NORM_STATS} has SHA-256 {sha}, expected the clean-only stats {NORM_SHA256}"

    text = src_yaml.read_text()
    # key-specific rewrites first, then generic prefix rewrites for every other absolute path
    text, n = re.subn(r"(?m)^(\s*norm_stats_file:).*$", rf"\1 {NORM_STATS}", text)
    assert n == 1, "data.norm_stats_file not found exactly once"
    has_dino = re.search(r"(?m)^\s*dino_fusion_path:\s*\S", text) and not re.search(
        r"(?m)^\s*dino_fusion_path:\s*(null|None|~)\s*$", text)
    if has_dino:
        text = re.sub(r"(?m)^(\s*dino_fusion_path:).*$", rf"\1 {DINO_CONFIG_DIR}", text)
    # continuation runs were initialized from another run's hf_ckpt (model.config_path / model_path); that directory
    # does not exist elsewhere, and the evaluated hf_ckpt has the same architecture config, so point them at it
    text = re.sub(r"(?m)^(\s*(?:config_path|model_path):)\s*/\S*/runs/\S*$", rf"\1 {ckpt}", text)
    text = re.sub(r"(?<=[\s'\"])/\S*?/ckpts/", f"{os.path.abspath(args.ckpt_dir)}/", text)
    text = re.sub(r"(?<=[\s'\"])/\S*?/LingBotRoboTwin/", f"{HERE}/", text)

    out_run = args.stage / run.name
    (out_run / "checkpoints" / step).mkdir(parents=True, exist_ok=True)
    (out_run / "lingbotvla_cli.yaml").write_text(text)
    link = out_run / "checkpoints" / step / "hf_ckpt"
    if link.is_symlink() or link.exists():
        assert link.is_symlink() and link.resolve() == ckpt.resolve(), f"{link} exists and is not a link to {ckpt}"
    else:
        os.symlink(ckpt, link)

    tok = re.search(r"(?m)^\s*tokenizer_path:\s*(\S+)", text)
    missing = [p for p in [tok.group(1) if tok else None] if p and not Path(p).exists()]
    assert not missing, f"not found: {missing} (download Qwen/Qwen3-VL-4B-Instruct into --ckpt_dir)"
    if has_dino:
        assert (DINO_CONFIG_DIR / "config.json").is_file(), f"missing {DINO_CONFIG_DIR}/config.json"
        index = (ckpt / "model.safetensors.index.json").read_text()
        assert "model.dino_fusion.dino." in index, "DINO-fusion config but no DINOv2 weights in the checkpoint"
    print(link)


if __name__ == "__main__":
    main()
