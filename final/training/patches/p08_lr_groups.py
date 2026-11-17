"""p08 (code_v4 worktree): lower learning rate for the VL backbone + model-only resume.

Official recipe (configs/vla/real_robot/real_robot.yaml, used by v2/v3): one LR (5e-5, Muon, constant) for everything
except the action expert's routed experts (x sqrt(32/4) = 2.83, use_moe_expert_lr). This patch adds
  LB_LR_SCALE_VIT  multiplier for the Qwen3-VL vision tower   (names containing .qwenvl.model.visual.)
  LB_LR_SCALE_LLM  multiplier for the Qwen3-VL language model (names containing .qwenvl.model.language_model.)
Defaults 1.0 = official. The action expert, its routed-expert scaling, the action/state projections and the
depth/video alignment heads keep the base LR. "_orig_mod." (torch.compile wrappers) is ignored when matching names;
setting a scale != 1 that matches no parameter aborts.

Changing the parameter groups makes the saved optimizer state incompatible (DCP's set_optimizer_state_dict copies the
saved groups' hyper-parameters, lr included, onto the new groups by FQN, and the LR scheduler restores its old
base_lrs), so with a new LR split the run must start with a fresh optimizer:
  LB_RESUME_MODEL_ONLY=1  --train.load_checkpoint_path <ckpt>  loads only the model weights from the DCP checkpoint;
  optimizer, LR scheduler, dataloader and step counter start fresh (step 0).
"""
from pathlib import Path

# Fixed 11/14 18:15: part 3 used to reuse the text read before part 2 and wrote it back, which dropped part 2
# (model-only resume) from code_v4. code_v4 is that broken tree; code_v4s (= code_v4 + part 2) is the correct one.
import os
root = Path(os.environ.get('P08_ROOT', str(Path.home() / 'lingbot_train/code_v4')))

# 1. per-module LR scale in the Muon/AdamW group split
p = root / 'lingbotvla/optim/optimizer.py'
s = p.read_text()
old = '''    """Bucket (param, name) pairs by their possibly MoE-scaled LR."""
    lr_to_params: Dict[float, List[torch.Tensor]] = {base_lr: []}
    for p, name in params_and_names:
        m = layer_re.search(name)
        lr_for_param = base_lr
        if m is not None:
            layer_idx = int(m.group(1))
            scale = layer_to_scale.get(layer_idx)
            if scale is not None:
                lr_for_param = base_lr * scale
        lr_to_params.setdefault(lr_for_param, []).append(p)
    return [{"params": ps, "lr": lr} for lr, ps in lr_to_params.items() if ps]'''
new = '''    """Bucket (param, name) pairs by their possibly MoE-scaled LR (p08: times the VL-backbone module scale)."""
    import os as _os
    module_scales = [(".qwenvl.model.visual.", float(_os.environ.get("LB_LR_SCALE_VIT", "1.0"))),
                     (".qwenvl.model.language_model.", float(_os.environ.get("LB_LR_SCALE_LLM", "1.0")))]
    matched = {pat: 0 for pat, _ in module_scales}
    lr_to_params: Dict[float, List[torch.Tensor]] = {base_lr: []}
    for p, name in params_and_names:
        clean = name.replace("_orig_mod.", "")
        m = layer_re.search(clean)
        lr_for_param = base_lr
        if m is not None:
            layer_idx = int(m.group(1))
            scale = layer_to_scale.get(layer_idx)
            if scale is not None:
                lr_for_param = base_lr * scale
        for pat, ms in module_scales:
            if pat in clean:
                matched[pat] += 1
                lr_for_param = lr_for_param * ms
        lr_to_params.setdefault(lr_for_param, []).append(p)
    for pat, ms in module_scales:
        if ms != 1.0 and params_and_names and matched[pat] == 0 and not _os.environ.get("LB_LR_SCALE_ALLOW_UNMATCHED"):
            raise RuntimeError(f"[p08] LR scale {ms} for '{pat}' matched no parameter in this group set")
    return [{"params": ps, "lr": lr} for lr, ps in lr_to_params.items() if ps]'''
assert s.count(old) == 1, s.count(old)
p.write_text(s.replace(old, new))

# 2. model-only resume
p = root / 'tasks/vla/train_lingbotvla.py'
s = p.read_text()
old = '''            state = {"model": model, "ema": None, "optimizer": optimizer, "extra_state": {}}  # cannot be None
            try:
                Checkpointer.load(cp, state, allow_partial_load=getattr(args.train, 'allow_partial_checkpoint', False))
                global_step = state["extra_state"]["global_step"]'''
new = '''            model_only = os.environ.get("LB_RESUME_MODEL_ONLY", "0") not in ("", "0")
            state = {"model": model, "ema": None, "optimizer": optimizer, "extra_state": {}}  # cannot be None
            if model_only:  # p08: weights only; fresh optimizer / LR scheduler / dataloader / step counter
                state = {"model": model, "ema": None}
            try:
                Checkpointer.load(cp, state, allow_partial_load=getattr(args.train, 'allow_partial_checkpoint', False))
                if model_only:
                    dist.barrier()
                    logger.info_rank0(f"[p08] Loaded MODEL WEIGHTS ONLY from {cp}: fresh optimizer, LR scheduler and dataloader, step 0")
                    loaded = True
                    break
                global_step = state["extra_state"]["global_step"]'''
assert s.count(old) == 1, s.count(old)
p.write_text(s.replace(old, new))

# 3. an explicitly requested checkpoint that fails to load must stop the run, not silently train from scratch
s = p.read_text()  # re-read: part 2 has just rewritten this file
old = '''        if not loaded:
            logger.info_rank0("Starting training from scratch. No valid checkpoint could be loaded.")'''
new = '''        if not loaded:
            if args.train.load_checkpoint_path:
                raise RuntimeError(f"[p08] could not load the requested checkpoint {args.train.load_checkpoint_path}: {last_err!r}")
            logger.info_rank0("Starting training from scratch. No valid checkpoint could be loaded.")'''
assert s.count(old) == 1, s.count(old)
p.write_text(s.replace(old, new))
print('patched p08', root)
