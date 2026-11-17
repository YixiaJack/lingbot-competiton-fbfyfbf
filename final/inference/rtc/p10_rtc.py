"""p10: training-time real-time chunking (RTC) for LingBot-VLA v2. Paper: Black, Ren, Equi, Levine 2025,
"Training-Time Action Conditioning for Efficient Real-Time Chunking", arXiv 2512.05964.

Training (LB_RTC_MAX_DELAY > 0; off by default):
  - Each sample gets a clean prefix of d ground-truth actions. d is 0 with probability LB_RTC_P0 (default 0.3, which
    keeps the no-prefix / synchronous path supervised); otherwise it is uniform in 1..LB_RTC_MAX_DELAY.
  - Prefix tokens get flow time 0. In this codebase x_t = t*noise + (1-t)*actions, so t = 0 is clean; the paper's
    convention is the opposite (tau = 1).
  - The loss covers only the tokens after the prefix, renormalized per sample.
Inference: sample_actions(..., prefix_actions=<normalized [B, T, D]>, prefix_len=d) holds the first d actions fixed
(clean, time 0) at every Euler step. With prefix_len=0 (the default) the code path is the original one.
Model changes: a per-token flow time ([B, T] instead of [B]) in embed_suffix, and AdaRMSNorm accepting a per-token
cond [B, 1+T, D] (the state token keeps the noisy tokens' time). No new parameters, so the optimizer state and the
checkpoints stay compatible both ways.

usage: python p10_rtc.py <code root>     (e.g. ~/lingbot_train/code_v4r; idempotent)"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).expanduser()
D = ROOT / 'lingbotvla/models/vla/lingbot_vla'
V1, V2 = D / 'modeling_lingbot_vla.py', D / 'modeling_lingbot_vla_v2.py'


def patch(path, edits, marker):
    s = path.read_text()
    if marker in s:
        print('already patched:', path)
        return
    for old, new, n in edits:
        c = s.count(old)
        assert c == n, (path.name, c, n, old[:80])
        s = s.replace(old, new)
    path.write_text(s)
    print('patched', path)


# ---------------------------------------------------------------- modeling_lingbot_vla.py (AdaRMSNorm, embed_suffix)
v1_edits = [
    ("        gamma = self.gamma(cond).unsqueeze(1)  # [B, 1, H]\n"
     "        beta  = self.beta(cond).unsqueeze(1)   # [B, 1, H]\n",
     "        gamma = self.gamma(cond)\n"
     "        beta  = self.beta(cond)\n"
     "        if cond.dim() == 2:  # [B, D] -> [B, 1, H]; p10 RTC: a per-token cond [B, N, D] gives [B, N, H]\n"
     "            gamma, beta = gamma.unsqueeze(1), beta.unsqueeze(1)\n", 2),
    ("        time_emb = create_sinusoidal_pos_embedding( # 1, 1024\n"
     "            timestep, # torch.Size([1]))\n",
     "        per_token_time = timestep.dim() == 2  # p10 RTC: one flow time per action token, [B, T]\n"
     "        time_emb = create_sinusoidal_pos_embedding( # 1, 1024\n"
     "            timestep.reshape(-1) if per_token_time else timestep, # torch.Size([1]))\n", 1),
    ("        time_emb_ori = time_emb\n",
     "        time_emb_ori = time_emb\n"
     "        if per_token_time:  # p10 RTC\n"
     "            time_emb = time_emb.view(timestep.shape[0], timestep.shape[1], -1)  # [B, T, D]\n"
     "            # adaLN cond per suffix token [state, a_0 .. a_T-1]. The state token keeps the noisy tokens' time;\n"
     "            # prefix tokens are clean (time 0), so that time is the row max.\n"
     "            state_time_emb = create_sinusoidal_pos_embedding(\n"
     "                timestep.max(dim=1).values, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device,\n"
     "            ).type(dtype=dtype)\n"
     "            time_emb_ori = torch.cat([state_time_emb[:, None], time_emb], dim=1)  # [B, 1+T, D]\n", 1),
    ('        time_emb = einops.repeat(time_emb, "b d -> b n d", n=action_emb.shape[1])',
     '        if not per_token_time:  # p10 RTC: per-token embeddings are already [B, T, D]\n'
     '            time_emb = einops.repeat(time_emb, "b d -> b n d", n=action_emb.shape[1])', 1),
]

# ---------------------------------------------------------------- modeling_lingbot_vla_v2.py (forward, sample_actions)
helpers = '''
import os as _p10_os

# p10: training-time real-time chunking (see experiments/train_real/patches/p10_rtc.py in the notes repo).
_P10_RTC_MAX_DELAY = int(_p10_os.environ.get("LB_RTC_MAX_DELAY", "0"))
_P10_RTC_P0 = float(_p10_os.environ.get("LB_RTC_P0", "0.3"))
if _P10_RTC_MAX_DELAY > 0:
    print(f"[p10] training-time RTC on: LB_RTC_MAX_DELAY={_P10_RTC_MAX_DELAY} LB_RTC_P0={_P10_RTC_P0}", flush=True)


def _p10_rtc_prefix_mask(bsize, horizon, device, max_delay=None, p0=None):
    """[B, T] bool, True on each sample's clean prefix: d = 0 with probability p0, else uniform in 1..max_delay."""
    max_delay = _P10_RTC_MAX_DELAY if max_delay is None else max_delay
    p0 = _P10_RTC_P0 if p0 is None else p0
    d = torch.randint(1, max_delay + 1, (bsize,), device=device)
    d = torch.where(torch.rand(bsize, device=device) < p0, torch.zeros_like(d), d)
    return torch.arange(horizon, device=device)[None, :] < d[:, None]


def _p10_rtc_mask_losses(losses, prefix_mask):
    """Zero the prefix tokens' loss and rescale each sample so its mean is taken over its postfix tokens."""
    keep = (~prefix_mask).to(losses.dtype)
    scale = keep.shape[1] / keep.sum(dim=1, keepdim=True).clamp(min=1)
    return losses * (keep * scale)[:, :, None]
'''

v2_edits = [
    ("from lingbotvla.ops.triton_moe_loss import triton_sequence_wise_balance_loss\n",
     "from lingbotvla.ops.triton_moe_loss import triton_sequence_wise_balance_loss\n" + helpers, 1),
    ("        time_expanded = time[:, None, None]\n"
     "        x_t = time_expanded * noise + (1 - time_expanded) * actions\n"
     "        u_t = noise - actions\n",
     "        time_expanded = time[:, None, None]\n"
     "        x_t = time_expanded * noise + (1 - time_expanded) * actions\n"
     "        u_t = noise - actions\n"
     "        rtc_prefix = None\n"
     "        suffix_time = time\n"
     "        if self.training and _P10_RTC_MAX_DELAY > 0:  # p10 RTC: clean ground-truth prefix, time 0\n"
     "            rtc_prefix = _p10_rtc_prefix_mask(actions.shape[0], actions.shape[1], device)\n"
     "            suffix_time = torch.where(rtc_prefix, torch.zeros_like(time)[:, None], time[:, None])  # [B, T]\n"
     "            x_t = suffix_time[:, :, None] * noise + (1 - suffix_time[:, :, None]) * actions\n", 1),
    ("self.embed_suffix(\n            state, x_t, time\n        )",
     "self.embed_suffix(\n            state, x_t, suffix_time\n        )", 1),
    ("        seq_wise_loss, router_z_loss, moe_metrics = self._moe_losses_and_metrics(\n"
     "            router_logits_list, losses\n        )",
     "        if rtc_prefix is not None:  # p10 RTC: no loss on the given prefix\n"
     "            losses = _p10_rtc_mask_losses(losses, rtc_prefix)\n"
     "        seq_wise_loss, router_z_loss, moe_metrics = self._moe_losses_and_metrics(\n"
     "            router_logits_list, losses\n        )", 1),
    ('        noise=None,\n        image_grid_thw=None,\n    ) -> Tensor:\n'
     '        """Do a full Qwen3-VL inference forward and compute the action."""',
     '        noise=None,\n        image_grid_thw=None,\n        prefix_actions=None,\n        prefix_len=0,\n    ) -> Tensor:\n'
     '        """Do a full Qwen3-VL inference forward and compute the action.\n'
     '        p10 RTC: prefix_actions [B, T, D] (normalized, model space) with prefix_len d > 0 holds the first d actions."""', 1),
    ("        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)\n"
     "        x_t = noise\n"
     "        time = torch.tensor(1.0, dtype=dtype, device=device)\n",
     "        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)\n"
     "        x_t = noise\n"
     "        time = torch.tensor(1.0, dtype=dtype, device=device)\n"
     "        rtc_d = int(prefix_len or 0)  # p10 RTC\n"
     "        if rtc_d > 0:\n"
     "            rtc_prefix = prefix_actions[:, :rtc_d].to(device=device, dtype=x_t.dtype)\n"
     "            x_t = x_t.clone()\n"
     "            x_t[:, :rtc_d] = rtc_prefix\n", 1),
    ("            expanded_time = time.expand(bsize)\n",
     "            expanded_time = time.expand(bsize)\n"
     "            if rtc_d > 0:  # p10 RTC: prefix tokens stay clean (time 0)\n"
     "                expanded_time = expanded_time[:, None].repeat(1, x_t.shape[1])\n"
     "                expanded_time[:, :rtc_d] = 0\n", 1),
    ("            x_t += dt * v_t\n            time += dt\n",
     "            x_t += dt * v_t\n"
     "            if rtc_d > 0:  # p10 RTC\n"
     "                x_t[:, :rtc_d] = rtc_prefix\n"
     "            time += dt\n", 1),
]

patch(V1, v1_edits, 'p10 RTC')
patch(V2, v2_edits, '_p10_rtc_prefix_mask')
