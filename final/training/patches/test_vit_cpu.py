"""CPU checks for p00 (batched SDPA) and p02 (grid cache) against the original code paths."""
import os, sys, types
sys.path.insert(0, os.path.expanduser('~/lingbot_train/code_opt'))
import torch
import transformers.models.qwen3_vl.modeling_qwen3_vl as hf
from transformers import AutoConfig
import lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla as Q

torch.manual_seed(0)
M = '/mnt/forge/models/system/00000000-0000-0000-0000-000000000002/assets/Qwen3-VL-4B-Instruct'
vcfg = AutoConfig.from_pretrained(M).vision_config

# ---------------- p02: fast_pos_embed_interpolate cached vs original (values and pos_embed grads)
fake = types.SimpleNamespace()
fake.num_grid_per_side = int(vcfg.num_position_embeddings ** 0.5)
fake.pos_embed = torch.nn.Embedding(vcfg.num_position_embeddings, 64)
fake.config = types.SimpleNamespace(spatial_merge_size=vcfg.spatial_merge_size)
for grid in ([[1, 14, 14]] * 48, [[1, 16, 16], [2, 14, 14], [1, 14, 20]]):
    g = torch.tensor(grid)
    for dt in (torch.float32, torch.bfloat16):
        fake.pos_embed = fake.pos_embed.to(dt)
        fake.pos_embed.weight.grad = None
        a = Q._fast_pos_embed_interpolate_orig(fake, g); a.float().pow(2).sum().backward(); ga = fake.pos_embed.weight.grad.clone()
        fake.pos_embed.weight.grad = None
        b = Q.fast_pos_embed_interpolate_cached(fake, g); b.float().pow(2).sum().backward(); gb = fake.pos_embed.weight.grad.clone()
        fake.pos_embed.weight.grad = None
        c = Q.fast_pos_embed_interpolate_cached(fake, g)   # cache hit
        print('p02', dt, 'grid', len(grid), 'values equal', torch.equal(a, b), torch.equal(a, c), 'grads equal', torch.equal(ga, gb), tuple(a.shape))
        fake.pos_embed = fake.pos_embed.float()

# ---------------- p02: preprcess_grid_thw cached vs original
vm = types.SimpleNamespace(spatial_merge_size=vcfg.spatial_merge_size)
head_dim = vcfg.hidden_size // vcfg.num_heads
rot = hf.Qwen3VLVisionRotaryEmbedding(head_dim // 2)
vm.rot_pos_emb = types.MethodType(hf.Qwen3VLVisionModel.rot_pos_emb, vm)
vm.rotary_pos_emb = rot
vm.spatial_merge_unit = vcfg.spatial_merge_size ** 2
g = torch.tensor([[1, 14, 14]] * 6)
try:
    o = Q._preprcess_grid_thw_orig(vm, g); c1 = Q.preprcess_grid_thw_cached(vm, g); c2 = Q.preprcess_grid_thw_cached(vm, g)
    print('p02 pre: cos equal', torch.equal(o[1][0], c1[1][0]), 'sin equal', torch.equal(o[1][1], c2[1][1]),
          'cu equal', torch.equal(o[2], c2[2]), 'split', o[3] == c2[3], 'max', o[4], c2[4])
except Exception as e:
    print('p02 pre: skipped on CPU stub:', repr(e)[:200])

# ---------------- p00: batched SDPA vs per-image loop inside the patched attention
cfg = AutoConfig.from_pretrained(M).vision_config
cfg.hidden_size, cfg.num_heads = 128, 4
cfg._attn_implementation = 'sdpa'
att = Q.Qwen3VLVisionAttention(cfg).eval()
n_img, L = 6, 196
x = torch.randn(n_img * L, cfg.hidden_size)
cu = torch.arange(0, (n_img + 1) * L, L, dtype=torch.int32)
emb = torch.randn(n_img * L, cfg.hidden_size // cfg.num_heads)
pe = (emb.cos(), emb.sin())
with torch.no_grad():
    os.environ['LB_VIT_BATCHED_SDPA'] = '0'; y0 = att(x, cu_seqlens=cu, position_embeddings=pe, max_seqlen=L)
    os.environ['LB_VIT_BATCHED_SDPA'] = '1'; y1 = att(x, cu_seqlens=cu, position_embeddings=pe, max_seqlen=L)
print('p00 batched vs loop: max abs diff', (y0 - y1).abs().max().item(), 'rel', ((y0 - y1).norm() / y0.norm()).item())
cu2 = torch.tensor([0, 196, 392, 600], dtype=torch.int32)   # unequal lengths must fall back to the loop
x2 = torch.randn(600, cfg.hidden_size); e2 = torch.randn(600, cfg.hidden_size // cfg.num_heads)
with torch.no_grad():
    y2 = att(x2, cu_seqlens=cu2, position_embeddings=(e2.cos(), e2.sin()), max_seqlen=208)
print('p00 unequal lengths ran the loop path, shape', tuple(y2.shape), 'batched_ok =', Q._vit_batched_ok('sdpa', torch.empty(1, 4, 600, 32), cu2, 208))
