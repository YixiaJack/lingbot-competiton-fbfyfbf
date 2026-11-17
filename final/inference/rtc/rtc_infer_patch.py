"""Inference side of async + RTC on the 5080 (lingbot_infer). Pairs with training-time RTC: patch p10 in
experiments/train_real/patches/p10_rtc.py, used by v4 from overall step 3000 on.

The official server already does the async part: async_pipeline, a staged next chunk, skipping the steps that elapsed
during inference, and handing rtc_prefix_len plus the remaining actions of the executing chunk to policy.preprocess.
Only the LingBot adapter refused it. This patch:
  1. model (p10): sample_actions(prefix_actions, prefix_len) holds the first d actions fixed at every Euler step;
  2. adapter (deploy/policy_adapter/policy_adapter.py, LingBotVLA2Policy):
     - validate_rtc checks the config and decides whether the loaded checkpoint is RTC-trained (rtc_trained.json in
       hf_ckpt, or its path listed in ~/rtc/allow.txt). It is not refused otherwise: the run continues as plain async
       with no prefix, and a warning is logged;
     - preprocess attaches the first d remaining absolute actions as "__rtc_prefix__";
  3. native runtime (code/lingbot-vla-v2/deploy/lingbot_vla_v2_policy.py):
     - converts that prefix into the model's normalized, state-relative, padded action space by inverting the official
       FeatureTransform.unapply, which is affine per dimension, from one probe;
     - drops the prefix if the round trip does not reproduce it;
     - passes it to sample_actions and logs one line per chunk (d, round trip, junction step vs median step).
With prefix_len 0 (sync config, or the first chunk of a session) every code path is the original one.
usage: python rtc_infer_patch.py <inference root, e.g. ~/workspace/lingbot/inference>   (idempotent; backs up files)"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1]).expanduser()
ADAPTER = ROOT / 'code/deploy/policy_adapter/policy_adapter.py'
NATIVE = ROOT / 'code/lingbot-vla-v2/deploy/lingbot_vla_v2_policy.py'
P10 = Path(__file__).resolve().parent / 'p10_rtc.py'
BACKUP = Path.home() / 'rtc/backup'
stamp = time.strftime('%Y%m%d_%H%M%S')


def patch(path, edits, marker):
    s = path.read_text()
    if marker in s:
        print('already patched:', path)
        return
    BACKUP.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, BACKUP / f'{path.name}.orig_{stamp}')
    for old, new, n in edits:
        c = s.count(old)
        assert c == n, (path.name, c, n, old[:90])
        s = s.replace(old, new)
    path.write_text(s)
    print('patched', path)


# ---------------------------------------------------------------- 1. model
model_dir = ROOT / 'code/lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla'
for f in ('modeling_lingbot_vla.py', 'modeling_lingbot_vla_v2.py'):
    if 'p10 RTC' not in (model_dir / f).read_text() and '_p10_rtc_prefix_mask' not in (model_dir / f).read_text():
        BACKUP.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_dir / f, BACKUP / f'{f}.orig_{stamp}')
subprocess.run([sys.executable, str(P10), str(ROOT / 'code/lingbot-vla-v2')], check=True)

# ---------------------------------------------------------------- 2. adapter
adapter_helpers = '''
import logging as _rtc_logging

# Team patch (p10 RTC, experiments/async_rtc): async + real-time chunking for RTC-trained checkpoints.
_RTC_MAX_PREFIX = 24   # largest prefix the RTC-trained checkpoints saw in training (LB_RTC_MAX_DELAY)
_rtc_log = _rtc_logging.getLogger("lingbot_infer")


def _rtc_ckpt_ok(model_path) -> tuple[bool, str]:
    path = Path(str(model_path))
    if (path / "rtc_trained.json").is_file():
        return True, "rtc_trained.json"
    allow = Path.home() / "rtc" / "allow.txt"
    if allow.is_file():
        for line in allow.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and line in str(path):
                return True, f"allow.txt: {line}"
    return False, "no rtc_trained.json and not in ~/rtc/allow.txt"
'''

adapter_methods = '''
    def validate_rtc(self, *, refill_threshold: int, action_hz: float) -> None:
        # Team patch (p10 RTC): config errors refuse START; a checkpoint without RTC training runs plain async.
        if refill_threshold > _RTC_MAX_PREFIX:
            raise ValueError(f"RTC: refill_threshold {refill_threshold} > trained prefix maximum {_RTC_MAX_PREFIX}")
        if self._use_length < 2 * refill_threshold:
            raise ValueError(f"RTC: use_length {self._use_length} must be >= 2 * refill_threshold {refill_threshold}")
        self._rtc_ok, why = _rtc_ckpt_ok(self.config.model_path)
        if self._rtc_ok:
            _rtc_log.info("RTC on: refill_threshold=%s use_length=%s action_hz=%s (%s)",
                          refill_threshold, self._use_length, action_hz, why)
        else:
            _rtc_log.warning("RTC requested but the checkpoint is not marked RTC-trained (%s): async WITHOUT prefix", why)

    def _rtc_prefix(self, rtc_prefix_len, prev_action_chunk):
        d = int(rtc_prefix_len or 0)
        if d <= 0 or prev_action_chunk is None or not getattr(self, "_rtc_ok", False):
            return None
        d = min(d, prev_action_chunk.chunk_length, _RTC_MAX_PREFIX)
        acts = prev_action_chunk.actions
        arm = np.concatenate([np.asarray(acts[ACTION_KEYS[f"{s}_arm.position"]], dtype=np.float32)[:d]
                              for s in self._arm_sides], axis=1)
        effector = np.concatenate([np.asarray(acts[ACTION_KEYS[f"{s}_gripper.position"]], dtype=np.float32)[:d]
                                   for s in self._arm_sides], axis=1)
        return {"d": d, "action.arm.position": arm, "action.effector.position": effector}
'''

adapter_edits = [
    ("from utils.canonical_keys import ACTION_KEYS, STATE_KEYS\n",
     "from utils.canonical_keys import ACTION_KEYS, STATE_KEYS\n" + adapter_helpers, 1),
    ('    def _reset_native(self) -> None:\n        self._native.reset(robo_name="agilex_cobot_magic")\n',
     '    def _reset_native(self) -> None:\n        self._native.reset(robo_name="agilex_cobot_magic")\n' + adapter_methods, 1),
    ('        if rtc_prefix_len is not None or prev_action_chunk is not None:\n'
     '            raise ValueError("The official LingBot VLA v2 runtime does not support RTC")\n',
     '        rtc = None\n'
     '        if rtc_prefix_len is not None or prev_action_chunk is not None:\n'
     '            # Team patch (p10 RTC): the executing chunk\'s remaining actions become the fixed prefix\n'
     '            rtc = self._rtc_prefix(rtc_prefix_len, prev_action_chunk)\n', 1),
    ('        return observation\n\n    def postprocess(',
     '        if rtc is not None:\n'
     '            observation["__rtc_prefix__"] = rtc\n'
     '        return observation\n\n    def postprocess(', 1),
]

# ---------------------------------------------------------------- 3. native runtime
native_methods = '''
    def _rtc_prefix_kwargs(self, transformed, rtc):
        """Team patch (p10 RTC): absolute prefix actions -> the model's normalized, state-relative, padded action space.
        FeatureTransform.unapply (model space -> absolute) is affine per dimension (unnormalize, + state for relative
        joints, dims selected by the joint masks), so it is inverted from one probe of D + 1 rows; the result must
        reproduce the prefix through unapply, otherwise no prefix is used."""
        import logging
        log = logging.getLogger("lingbot_infer")
        keys = list(self.action_key)
        T, D = int(transformed["actions"].shape[-2]), int(transformed["actions"].shape[-1])
        d = min(int(rtc["d"]), T)

        def unap(norm):
            single = dict(transformed)
            single["actions"] = norm.to(dtype=torch.float32, device="cpu")
            if "state" in single:
                single["state"] = single["state"].to(dtype=torch.float32)
            out = self.vla.feature_transform.unapply(single)
            return np.concatenate([np.asarray(out[k].float().cpu().numpy() if torch.is_tensor(out[k]) else out[k],
                                              dtype=np.float64).reshape(norm.shape[0], -1) for k in keys], axis=1)

        probe = torch.zeros(D + 1, D)
        probe[1:] = torch.eye(D)
        a = unap(probe)
        base, resp = a[0], a[1:] - a[0]            # resp[j, k]: effect of model dim j on absolute dim k
        nz = np.abs(resp) > 1e-9
        if not np.all(nz.sum(axis=0) == 1):
            log.warning("RTC: unapply is not one model dim per action dim (%s); no prefix", nz.sum(axis=0).tolist())
            return {}
        jk = nz.argmax(axis=0)
        scale = resp[jk, np.arange(resp.shape[1])]
        target = np.concatenate([np.asarray(rtc[k], dtype=np.float64)[:d].reshape(d, -1) for k in keys], axis=1)
        norm = np.zeros((T, D), dtype=np.float64)
        norm[:d, jk] = (target - base) / scale
        err = float(np.abs(unap(torch.from_numpy(norm).float())[:d] - target).max())
        if not np.isfinite(err) or err > 1e-3:
            log.warning("RTC: prefix round trip error %.2e; no prefix", err)
            return {}
        self._rtc_last = {"d": d, "err": err, "target": target}
        return {"prefix_actions": torch.from_numpy(norm).float()[None], "prefix_len": d}

    def _rtc_log_junction(self, unnormalized_actions):
        import logging
        try:
            last = self._rtc_last
            d, keys = last["d"], list(self.action_key)
            new = np.concatenate([np.asarray(unnormalized_actions[k][0], dtype=np.float64).reshape(
                np.asarray(unnormalized_actions[k][0]).shape[0], -1) for k in keys], axis=1)
            kept = float(np.abs(new[:d] - last["target"]).max())
            parts = []
            for k in keys:   # per action group (arm joints, grippers): step at the junction vs the median step after it
                a = np.asarray(unnormalized_actions[k][0], dtype=np.float64)
                st = np.abs(np.diff(a.reshape(a.shape[0], -1), axis=0)).max(axis=1)
                parts.append(f"{k.split('.')[1]} {st[d - 1]:.4f}/{float(np.median(st[d:])):.4f}")
            logging.getLogger("lingbot_infer").info(
                "RTC chunk: d=%d roundtrip=%.1e prefix_kept=%.1e junction/median step: %s",
                d, last["err"], kept, ", ".join(parts))
        except Exception:
            logging.getLogger("lingbot_infer").exception("RTC: junction log failed")
'''

native_edits = [
    ("        capture_time: bool = False,\n        sample_compile_fn: callable = None,\n    ) -> Tensor:\n",
     "        capture_time: bool = False,\n        sample_compile_fn: callable = None,\n"
     "        prefix_actions=None,\n        prefix_len=0,\n    ) -> Tensor:\n", 1),
    ("        else:\n"
     "            actions = sample_compile_fn(\n"
     "                            images.to(dtype=dtype, device=device),\n"
     "                            img_masks.to(device=device),\n"
     "                            lang_tokens.to(device=device),\n"
     "                            lang_masks.to(device=device),\n"
     "                            state.to(dtype=dtype, device=device),\n"
     "                            image_grid_thw=self._to_device_image_grid_thw(image_grid_thw, device),\n"
     "            )\n",
     "        else:\n"
     "            # team patch p10 RTC: prefix kwargs only when there is a prefix (otherwise the original call)\n"
     "            rtc_kwargs = {} if not prefix_len else {\n"
     "                \"prefix_actions\": prefix_actions.to(dtype=dtype, device=device), \"prefix_len\": int(prefix_len)}\n"
     "            actions = sample_compile_fn(\n"
     "                            images.to(dtype=dtype, device=device),\n"
     "                            img_masks.to(device=device),\n"
     "                            lang_tokens.to(device=device),\n"
     "                            lang_masks.to(device=device),\n"
     "                            state.to(dtype=dtype, device=device),\n"
     "                            image_grid_thw=self._to_device_image_grid_thw(image_grid_thw, device),\n"
     "                            **rtc_kwargs,\n"
     "            )\n", 1),
    ("    def _infer_batch(self, observations, return_normalized=False):\n",
     native_methods.lstrip('\n') + "\n    def _infer_batch(self, observations, return_normalized=False):\n", 1),
    ("        applied = [self._prepare_model_input(obs) for obs in observations] # bsize, dict{key }\n",
     "        # team patch p10 RTC: the adapter attaches the fixed prefix (absolute actions) as \"__rtc_prefix__\"\n"
     "        observations = [dict(obs) for obs in observations]\n"
     "        rtc = [obs.pop(\"__rtc_prefix__\", None) for obs in observations]\n"
     "        applied = [self._prepare_model_input(obs) for obs in observations] # bsize, dict{key }\n", 1),
    ("        actions = self.vla.sample_actions_batch(\n"
     "            batch_observation,\n"
     "            self.use_bf16,\n"
     "            self.use_compile,\n"
     "            capture_time=False,\n"
     "            sample_compile_fn = self.sample_actions_fn,\n"
     "        )\n"
     "        \n"
     "        unnormalized_actions = self._unapply_batched_actions(applied, actions)\n",
     "        rtc_kwargs = {}\n"
     "        if rtc[0] is not None and len(applied) == 1:\n"
     "            rtc_kwargs = self._rtc_prefix_kwargs(applied[0], rtc[0])\n"
     "        actions = self.vla.sample_actions_batch(\n"
     "            batch_observation,\n"
     "            self.use_bf16,\n"
     "            self.use_compile,\n"
     "            capture_time=False,\n"
     "            sample_compile_fn = self.sample_actions_fn,\n"
     "            **rtc_kwargs,\n"
     "        )\n"
     "        \n"
     "        unnormalized_actions = self._unapply_batched_actions(applied, actions)\n"
     "        if rtc_kwargs:\n"
     "            self._rtc_log_junction(unnormalized_actions)\n", 1),
]

patch(ADAPTER, adapter_edits, '_rtc_ckpt_ok')
patch(NATIVE, native_edits, '_rtc_prefix_kwargs')

# ---------------------------------------------------------------- 3b. config validation refused rtc_enabled outright
ARGS = ROOT / 'code/deploy/utils/arguments.py'
patch(ARGS, [(
    '            if self.scheduler.async_pipeline.rtc_enabled:\n'
    '                raise ValueError(\n'
    '                    "The official LingBot VLA v2 runtime does not support RTC"\n'
    '                )\n',
    '            # Team patch (p10 RTC): the patched LingBot adapter supports RTC and checks it in validate_rtc\n', 1)],
    'Team patch (p10 RTC)')

# ---------------------------------------------------------------- 4. configs: keep the sync one, write the RTC one
cfg_dir = ROOT / 'config'
sync, rtc_cfg = cfg_dir / 'inference.sync.yaml', cfg_dir / 'inference.rtc.yaml'
if not sync.exists():
    shutil.copy2(cfg_dir / 'inference.yaml', sync)
s = sync.read_text()
old_ul, old_sched = '    use_length: 25\n', '  interpolation: hold\n'
assert s.count(old_ul) == 1 and s.count(old_sched) == 1 and 'async_pipeline' not in s
s = s.replace(old_ul, '    use_length: 44\n').replace(
    old_sched, old_sched + '  async_pipeline:   # team patch p10 RTC (experiments/async_rtc)\n'
                           '    enabled: true\n    refill_threshold: 20\n    rtc_enabled: true\n')
rtc_cfg.write_text(s)
print('wrote', rtc_cfg, '(sync config kept as', sync.name + ')')
