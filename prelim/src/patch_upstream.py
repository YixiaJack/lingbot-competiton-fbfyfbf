"""Apply this repo's (small, env-gated) changes to the upstream lingbot-vla-v2 checkout. Idempotent.

Each change is inert unless its environment variable is set, so the released recipe still runs unchanged:
  LINGBOT_AUG_CONFIG=<json>     transform.py: whitelist image augmentation (src/lingbot_aug.py) replaces upstream's
                                color-only augmentation (still requires --data.image_augment true)
  LINGBOT_SEEN_POOL=1           base_dataset.py: per-read instruction swap within the episode's seen pool
                                (meta/seen_instructions.json, src/make_seen_pool.py)
  LINGBOT_STATE_NOISE=<std>     utils.py: additive Gaussian noise on the normalized proprioceptive state (train only)
  LINGBOT_VISION_LR_SCALE=<x>   optimizer.py: vision-encoder (".visual.") param groups get lr * x
  (p_fourier in the aug config)  transform.py: whole-image FFT amplitude modulation, model input only (src/lingbot_aug.py)
  (p_wrist_drop in the aug config) transform.py: wrist-camera dropout -> that view gets the upstream "missing camera"
                                padding with mask False (train only; head camera never dropped)
  LINGBOT_TASK_SAMPLING=uniform multi_vla_dataset.py: each task drawn with probability 1/50 (src/lingbot_sampling.py)
  LINGBOT_FRAME_WEIGHTS=1       multi_vla_dataset.py: frames drawn by meta/frame_weights.npy (src/make_frame_weights.py)
src/lingbot_aug.py, src/lingbot_sampling.py must be importable (train.sbatch puts $WORK_DIR/src on PYTHONPATH).

  python src/patch_upstream.py $LINGBOT_DIR      # then: git -C $LINGBOT_DIR diff > patches/lingbot-vla-v2.diff
"""

import sys
from pathlib import Path

PATCHES = {
    "lingbotvla/data/vla_data/transform.py": [
        (
            None,  # append
            "\n\n# --- LingBotRoboTwin: whitelist-only augmentation, enabled by LINGBOT_AUG_CONFIG ---\n"
            "import os as _lbrt_os  # noqa: E402\n"
            "if _lbrt_os.environ.get(\"LINGBOT_AUG_CONFIG\"):\n"
            "    from lingbot_aug import sample_visual_augmentation_params, apply_visual_augmentation  # noqa: E402,F811\n",
        ),
        (
            "                pil_images.append(pil_img)\n            img_masks.append(False)\n\n    if isinstance(images[0], torch.Tensor):\n        images = torch.stack(images, dim=0)  # (n, c, h, w)\n",
            "                pil_images.append(pil_img)\n            img_masks.append(False)\n\n"
            "    # LingBotRoboTwin: wrist-camera dropout (train only): drawn per sample in lingbot_aug and replayed on the\n"
            "    # future frames; the view becomes upstream's missing-camera padding with mask False\n"
            "    _wrist_drop = (augment_params or {}).get(\"wrist_drop\") if train else None\n"
            "    if _wrist_drop:\n"
            "        for _i, _key in enumerate(image_keys):\n"
            "            if _i < len(images) and any(f\"wrist_{_side}\" in _key for _side in _wrist_drop):\n"
            "                images[_i] = torch.full_like(images[_i], -1.0) if isinstance(images[_i], torch.Tensor) else np.zeros_like(images[_i])\n"
            "                img_masks[_i] = False\n"
            "                if use_depth_align:\n"
            "                    pil_images[_i] = np.zeros_like(pil_images[_i])\n"
            "    if isinstance(images[0], torch.Tensor):\n        images = torch.stack(images, dim=0)  # (n, c, h, w)\n",
        ),
        (
            "        if use_depth_align:\n            pil_image_dict[key] = img.cpu().numpy()\n\n        if image_processor is None:\n",
            "        if use_depth_align:\n            pil_image_dict[key] = img.cpu().numpy()\n\n"
            "        # LingBotRoboTwin: Fourier amplitude modulation (p_fourier in the aug config), model input only: the\n"
            "        # frozen teachers' image was taken just above; per-camera params are reused on the future frames\n"
            "        if train and (augment_params or {}).get(\"fourier\"):\n"
            "            from lingbot_aug import apply_fourier\n"
            "            img = apply_fourier(img, augment_params[\"fourier\"], key)\n\n"
            "        if image_processor is None:\n",
        ),
    ],
    "lingbotvla/data/vla_data/base_dataset.py": [
        (
            '        item["task"] = _get_task_name(self.meta.tasks, task_idx)\n',
            '        item["task"] = _get_task_name(self.meta.tasks, task_idx)\n'
            "        # LingBotRoboTwin: swap within this episode's seen-instruction pool (rule 2.1)\n"
            '        if os.environ.get("LINGBOT_SEEN_POOL") == "1":\n'
            '            if not hasattr(self, "_seen_pool"):\n'
            '                pool_file = self.root / "meta" / "seen_instructions.json"\n'
            "                self._seen_pool = json.loads(pool_file.read_text()) if pool_file.exists() else {}\n"
            "            choices = self._seen_pool.get(str(ep_idx))\n"
            "            if choices:\n"
            '                item["task"] = choices[int(torch.randint(len(choices), ()))]\n',
        ),
        ("import os\n", "import os\nimport json  # LingBotRoboTwin\n"),
    ],
    "lingbotvla/data/vla_data/utils.py": [
        (
            "        state = prepare_state(batch_dict, self.model_config.max_state_dim) \n",
            "        state = prepare_state(batch_dict, self.model_config.max_state_dim) \n"
            "        # LingBotRoboTwin: proprioception noise (rule 2.1), train only, real (non-padded) dims only\n"
            '        _state_noise = float(os.environ.get("LINGBOT_STATE_NOISE", "0"))\n'
            "        if _state_noise > 0 and self.image_augment and not policy_eval:\n"
            "            state = state + torch.randn_like(state) * _state_noise * (state != 0)\n",
        ),
    ],
    "lingbotvla/data/vla_data/multi_vla_dataset.py": [
        (
            "        # Determine which dataset to get an item from based on the index.\n",
            "        # LingBotRoboTwin: task-balanced / motion-weighted sampling (rule 2.1: reweighting by data statistics)\n"
            "        if os.environ.get(\"LINGBOT_TASK_SAMPLING\") or os.environ.get(\"LINGBOT_FRAME_WEIGHTS\") == \"1\":\n"
            "            from lingbot_sampling import remap_index\n"
            "            idx = remap_index(self, idx)\n"
            "        # Determine which dataset to get an item from based on the index.\n",
        ),
        ("import numpy as np\n", "import os  # LingBotRoboTwin\nimport numpy as np\n"),
    ],
    "lingbotvla/optim/optimizer.py": [
        (
            "            if scale is not None:\n                lr_for_param = base_lr * scale\n",
            "            if scale is not None:\n                lr_for_param = base_lr * scale\n"
            "        # LingBotRoboTwin: separate vision-encoder LR multiplier\n"
            '        if ".visual." in name:\n'
            '            lr_for_param = lr_for_param * float(os.environ.get("LINGBOT_VISION_LR_SCALE", "1"))\n',
        ),
        ("import re\n", "import os  # LingBotRoboTwin\nimport re\n"),
    ],
}


def main():
    root = Path(sys.argv[1])
    for rel, edits in PATCHES.items():
        path = root / rel
        code = path.read_text()
        changed = False
        for old, new in edits:
            if new in code:  # this edit is already applied
                continue
            if old is None:
                code += new
            else:
                assert code.count(old) == 1, f"{rel}: anchor not found exactly once: {old!r}"
                code = code.replace(old, new)
            changed = True
        if changed:
            path.write_text(code)
        print("patched" if changed else "already patched", rel)


if __name__ == "__main__":
    main()
