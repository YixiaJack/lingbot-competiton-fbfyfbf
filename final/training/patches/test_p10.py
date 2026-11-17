"""CPU unit test of p10 (training-time RTC) on a patched copy of lingbotvla. No weights, no GPU: the changed functions
are run on small random modules or stubs, and the unchanged paths are checked to be numerically identical.
usage: LB_RTC_MAX_DELAY=24 LB_RTC_P0=0.3 CUDA_VISIBLE_DEVICES= python test_p10.py <patched code root>"""
import os, sys
from types import SimpleNamespace as NS

sys.path.insert(0, sys.argv[1])
import torch
# lingbotvla's group-gemm kernels query the GPU at import time; the test itself never touches a GPU
torch.cuda.get_device_capability = lambda *a, **k: (8, 0)
torch.cuda.get_device_name = lambda *a, **k: 'cpu-test'
from torch import nn
from lingbotvla.models.vla.lingbot_vla import modeling_lingbot_vla as V1
from lingbotvla.models.vla.lingbot_vla import modeling_lingbot_vla_v2 as V2

assert V2._P10_RTC_MAX_DELAY == int(os.environ['LB_RTC_MAX_DELAY']), 'env not picked up'
torch.manual_seed(0)
B, T, A, W = 4, 50, 55, 64
ok = lambda name: print('PASS', name)

# 1. AdaRMSNorm / FixAdaRMSNorm: per-token cond equal to the per-sample cond gives the same output
for cls in (V1.AdaRMSNorm, V1.FixAdaRMSNorm):
    n = cls(W, 32)
    for p in (n.gamma.weight, n.gamma.bias, n.beta.weight, n.beta.bias):
        nn.init.normal_(p, std=0.1)
    h, c = torch.randn(B, T + 1, W), torch.randn(B, 32)
    o2, o3 = n(h, c), n(h, c[:, None].expand(B, T + 1, 32))
    assert o2.shape == o3.shape == h.shape and torch.equal(o2, o3), cls
    c3 = torch.randn(B, T + 1, 32)
    assert torch.allclose(n(h, c3)[:, 5], n(h[:, 5:6], c3[:, 5])[:, 0], atol=1e-6)
    ok(f'{cls.__name__}: [B,D] == broadcast [B,N,D]; per-token cond applied per token')

# 2. embed_suffix: per-token time equal everywhere == original; with a prefix only the prefix tokens change
stub = nn.Module()
stub.state_proj, stub.action_in_proj = nn.Linear(20, W), nn.Linear(A, W)
stub.action_time_mlp_in, stub.action_time_mlp_out = nn.Linear(2 * W, W), nn.Linear(W, W)
stub.config = NS(proj_width=W, action_fp32=False)
emb = V2.FlowMatchingV2.embed_suffix
state, x, t = torch.randn(B, 20), torch.randn(B, T, A), torch.rand(B) * 0.9 + 0.05
te1, e1, p1, a1 = emb(stub, state, x, t)
te2, e2, p2, a2 = emb(stub, state, x, t[:, None].expand(B, T).contiguous())
assert te1.shape == (B, W) and te2.shape == (B, T + 1, W)
assert torch.equal(e1, e2) and torch.equal(p1, p2) and torch.equal(a1, a2)
assert torch.allclose(te2, te1[:, None].expand(B, T + 1, W))
d = 18
tp = t[:, None].repeat(1, T); tp[:, :d] = 0
te3, e3, _, _ = emb(stub, state, x, tp)
assert torch.equal(e3[:, 1 + d:], e1[:, 1 + d:]) and torch.equal(e3[:, 0], e1[:, 0]) and not torch.allclose(e3[:, 1:1 + d], e1[:, 1:1 + d])
assert torch.allclose(te3[:, 0], te1) and torch.allclose(te3[:, 1 + d:], te1[:, None].expand(B, T - d, W))
ok('embed_suffix: per-token == original when times are equal; prefix changes only prefix tokens; state cond keeps t')

# 3. prefix sampling and loss masking
m = V2._p10_rtc_prefix_mask(20000, T, 'cpu')
dd = m.sum(1)
p0 = (dd == 0).float().mean().item()
assert abs(p0 - V2._P10_RTC_P0) < 0.02 and dd.max().item() == V2._P10_RTC_MAX_DELAY and torch.all(m[:, 0] | (dd == 0))
assert torch.equal(m, torch.arange(T)[None] < dd[:, None])
L = torch.rand(8, T, A)
mm = V2._p10_rtc_mask_losses(L, m[:8])
for i in range(8):
    k = int(dd[i])
    assert torch.all(mm[i, :k] == 0) and torch.allclose(mm[i].mean(), L[i, k:].mean(), atol=1e-6)
ok(f'prefix mask: P(d=0)={p0:.3f}, d max={dd.max().item()}, mean d|d>0={dd[dd > 0].float().mean():.1f}; '
   'masked mean == postfix mean')

# 4. FlowMatchingV2.forward with stubs: train mode gets the clean prefix, time 0 there, no loss there;
#    eval mode is the original path
rec = {}
f = nn.Module()
f.sample_time = lambda b, dev: torch.full((b,), 0.7)
f.embed_prefix = lambda *a, **k: (torch.zeros(B, 8, 16), torch.ones(B, 8, dtype=torch.bool), torch.zeros(B, 8, dtype=torch.bool), None, None, None)
def _suffix(st, x_t, tm):
    rec['x_t'], rec['time'] = x_t.clone(), tm.clone()
    return None, torch.zeros(B, T + 1, 16), torch.ones(B, T + 1, dtype=torch.bool), torch.zeros(B, T + 1, dtype=torch.bool)
f.embed_suffix = _suffix
f.block_future_depth_to_action = False
f._block_suffix_to_future_video_if_enabled_ = lambda m_, **k: m_
f._build_full_position_ids = lambda *a: None
f.qwenvl_with_expert = NS(forward=lambda **k: ((None, torch.randn(B, T + 1, 16)), None, []))
f.config = NS(align_params={}, n_action_steps=T, action_fp32=False, use_cache=False, adanorm_time=True)
f.action_out_proj = nn.Linear(16, A)
def _moe(rl, losses):
    rec['losses'] = losses
    return 0, 0, {}
f._moe_losses_and_metrics = _moe
acts, noise = torch.randn(B, T, A), torch.randn(B, T, A)
fw = V2.FlowMatchingV2.forward
torch.manual_seed(1)
f.train()
fw(f, None, None, None, None, torch.randn(B, 20), acts, noise=noise)
tm, xt, ls = rec['time'], rec['x_t'], rec['losses']
assert tm.shape == (B, T)
for i in range(B):
    k = int((tm[i] == 0).sum())
    assert torch.all(tm[i, k:] == 0.7) and torch.allclose(xt[i, :k], acts[i, :k]) and torch.all(ls[i, :k] == 0)
    assert torch.allclose(xt[i, k:], 0.7 * noise[i, k:] + 0.3 * acts[i, k:], atol=1e-6)
ks = [int((tm[i] == 0).sum()) for i in range(B)]
f.eval()
fw(f, None, None, None, None, torch.randn(B, 20), acts, noise=noise)
assert rec['time'].shape == (B,) and torch.allclose(rec['x_t'], 0.7 * noise + 0.3 * acts, atol=1e-6)
ok(f'forward: train mode prefix lengths {ks} are clean with time 0 and no loss; eval mode is the original path')

# 5. sample_actions with a stub velocity: prefix_len=0 is the original Euler loop; prefix_len=d keeps the prefix exactly
target = torch.randn(B, T, A)
calls = []
s = nn.Module()
s.config = NS(num_steps=10, n_action_steps=T, max_action_dim=A, use_cache=True)
s.embed_prefix = f.embed_prefix
s.qwenvl_with_expert = NS(forward=lambda **k: (None, 'kv', None))
def _pv(st, pad, kv, x_t, timestep, prefix_position_ids=None):
    calls.append(timestep.clone())
    tt = timestep if timestep.dim() == 2 else timestep[:, None].expand(B, T)
    return (x_t - target) / tt.clamp(min=0.05)[..., None]
s.predict_velocity = _pv
sa = V2.FlowMatchingV2.sample_actions
noise = torch.randn(B, T, A)
st = torch.randn(B, 20)
out0 = sa(s, None, None, None, None, st, noise=noise.clone())
x, tt, dt = noise.clone(), torch.tensor(1.0), torch.tensor(-0.1)
while tt >= -dt / 2:
    x += dt * _pv(None, None, None, x, tt.expand(B)); tt += dt
assert torch.equal(out0, x) and all(c.dim() == 1 for c in calls[:10])
calls.clear()
prefix = torch.randn(B, T, A)
out1 = sa(s, None, None, None, None, st, noise=noise.clone(), prefix_actions=prefix, prefix_len=d)
assert torch.equal(out1[:, :d], prefix[:, :d]) and len(calls) == 10
assert all(c.shape == (B, T) and torch.all(c[:, :d] == 0) and torch.all(c[:, d:] == c[0, d]) for c in calls)
assert torch.equal(out1[:, d:], out0[:, d:])   # the stub has no cross-token mixing: the postfix integration is unchanged
ok(f'sample_actions: prefix_len=0 == original loop (10 steps); prefix_len={d} keeps the prefix exactly, time 0 there')
print('ALL PASS')
