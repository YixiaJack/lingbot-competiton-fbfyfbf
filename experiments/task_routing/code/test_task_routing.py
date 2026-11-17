import json, os, tempfile, torch, torch.nn as nn
cfg = {"task_routers": 3, "rules": [[1, r"\bdrawer"], [2, r"(pour|empty|dump|bin\b|trash|balls|tilt)"], [0, r"\b(left|right)"]],
       "lora": {"layers": [0], "rank": 2, "alpha": 4, "sets": 3}}
f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); json.dump(cfg, f); f.close()
os.environ["LINGBOT_ROUTE_CONFIG"] = f.name
from lingbotvla.models.vla.lingbot_vla import task_routing as tr
assert tr.task_id_from_text("Move the cup to the left of the bowl") == 0
assert tr.task_id_from_text("open the drawer and put it inside") == 1
assert tr.task_id_from_text("grab the tabletrashbin and tilt") == 2
assert tr.task_id_from_text("hello") == -1
blk = nn.Module(); blk.gate = nn.Linear(8, 4, bias=False); tr.add_task_gates(blk)
with torch.no_grad():
    for t, g in enumerate(blk.task_gates): g.copy_(torch.randn(4, 8))
B, T = 4, 3; h = torch.randn(B * T, 8)
tr.set_task_ids(torch.tensor([0, 2, -1, 1]))
lg = tr.router_logits(blk, h, B, T)
ref = torch.stack([h[i*T:(i+1)*T] @ (blk.task_gates[t].T if t >= 0 else blk.gate.weight.T) for i, t in enumerate([0, 2, -1, 1])]).reshape(B*T, 4)
assert torch.allclose(lg, ref, atol=1e-5), (lg - ref).abs().max()
tr.set_task_ids(None); assert torch.allclose(tr.router_logits(blk, h, B, T), h @ blk.gate.weight.T, atol=1e-6)
lora = tr.KVLoRA(8, 6, 6, 2, 4, 3)
with torch.no_grad():
    for b in list(lora.k_B): b.copy_(torch.randn_like(b))
x = torch.randn(B, 5, 8); tr.set_task_ids(torch.tensor([0, 2, -1, 1]))
d = lora.delta_k(x)
for i, t in enumerate([0, 2, -1, 1]):
    want = torch.zeros(5, 6) if t < 0 else (x[i] @ lora.k_A[t].T) @ lora.k_B[t].T * 2.0
    assert torch.allclose(d[i], want, atol=1e-5), (i, t)
z = tr.KVLoRA(8, 6, 6, 2, 4, 1); assert torch.count_nonzero(z.delta_v(x)) == 0   # B = 0 at init
d.sum().backward(); assert lora.k_B[1].grad is not None and lora.k_B[1].grad.abs().sum() > 0
print("task_routing unit tests: all passed")
