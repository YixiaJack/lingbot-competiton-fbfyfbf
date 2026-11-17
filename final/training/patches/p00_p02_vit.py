"""Patches on lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py (code_opt branch).

p00 LB_VIT_BATCHED_SDPA (default 1): when every image in the ViT batch has the same number of patches (true here:
    all frames are resized to the same size), run the non-flash-attn path as ONE batched SDPA over [n_img, heads, L, d]
    instead of a Python loop of per-image SDPA calls after `lengths.tolist()` (the loop broke torch.compile and ran
    ~1100 tiny kernels per micro batch). Block-diagonal attention == batched attention, so the math is unchanged.
p02 LB_GRID_CACHE (default 1): cache the grid_thw-only metadata of the vision tower. fast_pos_embed_interpolate builds
    its bilinear indices / weights / merge permutation with Python loops and .tolist() on every call; preprcess_grid_thw
    recomputes the rotary cos/sin and calls .item(). Both depend only on grid_thw (and dtype/device), so they are built
    once per grid with the original code and reused. pos_embed (trainable) is still looked up on every call.
"""
from pathlib import Path

p = Path.home() / 'lingbot_train/code_opt/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py'
s = p.read_text()

# ---------------------------------------------------------------- p00: batched SDPA for equal-length images
old = '''        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = ['''
new = '''        elif _vit_batched_ok(attn_impl, query_states, cu_seqlens, max_seqlen):
            # all images have max_seqlen patches: [1, H, n*L, d] -> [n, H, L, d], one SDPA call (no host sync)
            n_img = cu_seqlens.shape[0] - 1
            q_b, k_b, v_b = (
                t.reshape(t.shape[1], n_img, max_seqlen, t.shape[-1]).transpose(0, 1)
                for t in (query_states, key_states, value_states)
            )
            attn_b, _ = attention_interface(
                self,
                q_b,
                k_b,
                v_b,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )
            # attention_interface returns [n, L, H, d]; the loop path concatenated per-image [1, L, H, d] on dim 1
            attn_output = attn_b.reshape(1, n_img * max_seqlen, attn_b.shape[-2], attn_b.shape[-1])
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = ['''
assert s.count(old) == 1, ('p00 anchor', s.count(old))
s = s.replace(old, new)

helper = '''

import os as _os


def _vit_batched_ok(attn_impl, query_states, cu_seqlens, max_seqlen):
    """True when the varlen batch is n images of exactly max_seqlen patches each (decided from shapes, no sync)."""
    if _os.environ.get("LB_VIT_BATCHED_SDPA", "1") != "1" or attn_impl not in ("sdpa", "eager"):
        return False
    if max_seqlen is None or cu_seqlens is None:
        return False
    n_img = cu_seqlens.shape[0] - 1
    return n_img > 0 and int(max_seqlen) * n_img == query_states.shape[2]

'''
anchor = '\n_VIT_FA2_OK = None'
assert s.count(anchor) == 1
s = s.replace(anchor, helper + anchor)

# ---------------------------------------------------------------- p02: grid metadata cache
cache_code = '''

_GRID_META = {}
_fast_pos_embed_interpolate_orig = hf_qwen3vl.Qwen3VLVisionModel.fast_pos_embed_interpolate
_preprcess_grid_thw_orig = preprcess_grid_thw


def _grid_key(grid_thw):
    return tuple(int(v) for v in grid_thw.reshape(-1).tolist())


@torch.compiler.disable
def preprcess_grid_thw_cached(self, grid_thw: torch.Tensor):
    if _os.environ.get("LB_GRID_CACHE", "1") != "1":
        return _preprcess_grid_thw_orig(self, grid_thw)
    key = ("pre", _grid_key(grid_thw), str(grid_thw.device))
    hit = _GRID_META.get(key)
    if hit is None:
        hit = _preprcess_grid_thw_orig(self, grid_thw)
        _GRID_META[key] = hit
    pos_embeds, position_embeddings, cu_seqlens, split_sizes, max_seqlen = hit
    return pos_embeds, position_embeddings, cu_seqlens, list(split_sizes), max_seqlen


def _build_pos_meta(self, grid_thw):
    """Indices, weights and merge permutation exactly as transformers 4.57 fast_pos_embed_interpolate builds them."""
    grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]
    for t, h, w in zip(grid_ts, grid_hs, grid_ws):
        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor
        base_h = h_idxs_floor * self.num_grid_per_side
        base_h_ceil = h_idxs_ceil * self.num_grid_per_side
        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]
        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]
        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())
    dev, dt = self.pos_embed.weight.device, self.pos_embed.weight.dtype
    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=dev)
    weight_tensor = torch.tensor(weight_list, dtype=dt, device=dev)
    # the permutation the original applies after summing the 4 corners: run its split/repeat/view/permute on row ids
    merge_size = self.config.spatial_merge_size
    rows = torch.arange(idx_tensor.shape[1], device=dev).unsqueeze(-1)
    hw = [int(h) * int(w) for h, w in zip(grid_hs, grid_ws)]
    out = []
    for r, t, h, w in zip(rows.split(hw), grid_ts, grid_hs, grid_ws):
        t, h, w = int(t), int(h), int(w)
        r = r.repeat(t, 1)
        r = r.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1).permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
        out.append(r)
    perm = torch.cat(out).squeeze(-1)
    return idx_tensor, weight_tensor, perm


@torch.compiler.disable
def fast_pos_embed_interpolate_cached(self, grid_thw):
    if _os.environ.get("LB_GRID_CACHE", "1") != "1":
        return _fast_pos_embed_interpolate_orig(self, grid_thw)
    w = self.pos_embed.weight
    key = ("pos", _grid_key(grid_thw), str(w.device), str(w.dtype))
    meta = _GRID_META.get(key)
    if meta is None:
        meta = _build_pos_meta(self, grid_thw)
        _GRID_META[key] = meta
    idx_tensor, weight_tensor, perm = meta
    pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
    return patch_pos_embeds.index_select(0, perm)

'''
anchor2 = '\n\ndef forward_without_grid_thw('
assert s.count(anchor2) == 1
s = s.replace(anchor2, cache_code + anchor2[1:])

old_reg = '''    hf_qwen3vl.Qwen3VLVisionModel.preprcess_grid_thw = preprcess_grid_thw'''
new_reg = '''    hf_qwen3vl.Qwen3VLVisionModel.preprcess_grid_thw = preprcess_grid_thw_cached
    hf_qwen3vl.Qwen3VLVisionModel.fast_pos_embed_interpolate = fast_pos_embed_interpolate_cached'''
assert s.count(old_reg) == 1
s = s.replace(old_reg, new_reg)
p.write_text(s)
print('patched p00+p02', p)
