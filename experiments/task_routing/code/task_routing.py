"""Task-specific routing for LingBot-VLA 2.0 (final-round pre-competition ablation).

Enabled only when LINGBOT_ROUTE_CONFIG points at a JSON file; without it every patched code path is the stock one.

  {
    "task_routers": 3,                              # extra routers per MoE layer (0 = none)
    "rules": [[1, "\\bdrawer\\b"], [2, "\\b(pour|empty|dump|bin)\\b"], [0, "\\b(left|right)\\b"]],
    "lora": {"layers": [2, 5, 8], "rank": 16, "alpha": 32, "sets": 1}   # optional; sets 1 = shared, 3 = per task
  }

task_id comes from rules applied in order to the instruction text (first match wins, case-insensitive); no match = -1.
Router: logits of task t come from task_gates[t], initialised as copies of the pretrained gate (by the init checkpoint);
task -1 uses the pretrained gate itself, which stays frozen. The 32 routed experts, the shared expert, attention and
projections are shared. LoRA: added to the K and V projections of the chosen Qwen3-VL text layers; B starts at zero,
so step 0 equals the base model. With sets > 1 the task_id of each sample picks its LoRA; task -1 gets no delta.

The task ids of the current forward pass live in CTX (set by the policy forward in training and by the deploy policy at
inference), so no model signature changes.
"""
import json
import math
import os
import re
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Ctx:
    ids: Optional[torch.Tensor] = None   # (B,) long, one task id per sample


CTX = _Ctx()
_CFG = None


def config() -> dict:
    """The routing config (cached); {} when routing is off."""
    global _CFG
    if _CFG is None:
        path = os.environ.get("LINGBOT_ROUTE_CONFIG", "")
        _CFG = json.loads(open(path).read()) if path else {}
        if _CFG:
            _CFG["_compiled"] = [(int(t), re.compile(p, re.I)) for t, p in _CFG.get("rules", [])]
    return _CFG


def enabled() -> bool:
    return bool(config())


def task_id_from_text(text: str) -> int:
    for tid, pat in config().get("_compiled", []):
        if pat.search(text or ""):
            return tid
    return -1


def task_ids_from_prompts(prompts) -> torch.Tensor:
    if isinstance(prompts, str):
        prompts = [prompts]
    return torch.tensor([task_id_from_text(p) for p in prompts], dtype=torch.long)


def set_task_ids(ids: Optional[torch.Tensor]):
    CTX.ids = ids


def set_from_prompts(prompts):
    """Deploy side: route each observation by its instruction; an unmatched one is printed (it uses the base router)."""
    ids = task_ids_from_prompts(prompts)
    if (ids < 0).any():
        print(f"task routing: no rule matched {[p for p, i in zip(prompts, ids.tolist()) if i < 0]}", flush=True)
    set_task_ids(ids)


def per_token(batch_size: int, seq_len: int, device) -> Optional[torch.Tensor]:
    """(batch_size * seq_len,) task id per token, or None when no ids are set."""
    ids = CTX.ids
    if ids is None:
        return None
    ids = ids.to(device=device, dtype=torch.long).reshape(-1)
    if ids.numel() == 1 and batch_size != 1:
        ids = ids.expand(batch_size)
    if ids.numel() != batch_size:
        raise RuntimeError(f"task routing: {ids.numel()} task ids for a batch of {batch_size}")
    return ids.repeat_interleave(seq_len)


def per_sample(batch_size: int, device) -> Optional[torch.Tensor]:
    t = per_token(batch_size, 1, device)
    return t


# ---------------------------------------------------------------- router
def add_task_gates(moe_block: nn.Module):
    n = int(config().get("task_routers", 0))
    if n <= 0:
        moe_block.task_gates = None
        return
    w = moe_block.gate.weight
    moe_block.task_gates = nn.ParameterList(
        [nn.Parameter(torch.zeros_like(w)) for _ in range(n)])   # real values come from the init checkpoint


def router_logits(moe_block: nn.Module, hidden_flat: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    """fp32 router logits (N, E); stock gate when routing is off or no ids are set."""
    gates = getattr(moe_block, "task_gates", None)
    tok = per_token(batch_size, seq_len, hidden_flat.device) if gates is not None else None
    if tok is None:
        return F.linear(hidden_flat.float(), moe_block.gate.weight.float())
    E = moe_block.gate.weight.shape[0]
    W = torch.cat([moe_block.gate.weight] + list(gates), dim=0).float()          # ((R+1)E, D)
    all_logits = F.linear(hidden_flat.float(), W).view(hidden_flat.shape[0], len(gates) + 1, E)
    idx = (tok + 1).clamp(min=0, max=len(gates)).view(-1, 1, 1).expand(-1, 1, E)
    return all_logits.gather(1, idx).squeeze(1)


# ---------------------------------------------------------------- VLM K/V LoRA
class KVLoRA(nn.Module):
    def __init__(self, d_in: int, d_k: int, d_v: int, rank: int, alpha: float, sets: int):
        super().__init__()
        self.sets, self.scale = sets, alpha / rank

        def mats(d_out):
            A = nn.ParameterList([nn.Parameter(torch.empty(rank, d_in)) for _ in range(sets)])
            B = nn.ParameterList([nn.Parameter(torch.zeros(d_out, rank)) for _ in range(sets)])
            for a in A:
                nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            return A, B

        self.k_A, self.k_B = mats(d_k)
        self.v_A, self.v_B = mats(d_v)

    def _delta(self, x: torch.Tensor, A: Sequence[nn.Parameter], B: Sequence[nn.Parameter]) -> torch.Tensor:
        if self.sets == 1:
            return (x @ A[0].to(x.dtype).t()) @ B[0].to(x.dtype).t() * self.scale
        ids = per_sample(x.shape[0], x.device)
        if ids is None:
            return torch.zeros(*x.shape[:-1], B[0].shape[0], dtype=x.dtype, device=x.device)
        onehot = F.one_hot((ids + 1).clamp(min=0, max=self.sets), self.sets + 1)[:, 1:].to(x.dtype)   # (B, sets)
        out = 0
        for s in range(self.sets):
            d = (x @ A[s].to(x.dtype).t()) @ B[s].to(x.dtype).t()
            out = out + d * onehot[:, s].view(-1, *([1] * (x.ndim - 1)))
        return out * self.scale

    def delta_k(self, x):
        return self._delta(x, self.k_A, self.k_B)

    def delta_v(self, x):
        return self._delta(x, self.v_A, self.v_B)


def install_kv_lora(language_model: nn.Module) -> List[str]:
    """Adds self_attn.kv_lora to the configured text layers. Returns the names of the new parameters' modules."""
    cfg = config().get("lora")
    if not cfg:
        return []
    done = []
    for i in cfg["layers"]:
        attn = language_model.layers[i].self_attn
        attn.kv_lora = KVLoRA(attn.k_proj.in_features, attn.k_proj.out_features, attn.v_proj.out_features,
                              int(cfg["rank"]), float(cfg["alpha"]), int(cfg.get("sets", 1)))
        done.append(f"layers.{i}.self_attn.kv_lora")
    return done


def lora_params(model: nn.Module):
    return [p for n, p in model.named_parameters() if ".kv_lora." in n]
