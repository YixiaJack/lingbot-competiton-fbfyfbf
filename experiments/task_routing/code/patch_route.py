"""Apply the task-routing patch to a LingBot checkout copy. Idempotent; every edit asserts its anchor once.

  python route/patch_route.py <checkout>

Never run it on a checkout that live jobs use (dev/lingbot-vla-v2-dino is the A/B's). Without LINGBOT_ROUTE_CONFIG the
patched code behaves exactly like the original.
"""
import shutil
import sys
from pathlib import Path

L = Path(sys.argv[1])
HERE = Path(__file__).resolve().parent
MARK = "# [route]"


def edit(rel, pairs):
    p = L / rel
    t = p.read_text()
    for old, new in pairs:
        if new in t:
            continue
        assert t.count(old) == 1, (rel, old[:80], t.count(old))
        t = t.replace(old, new)
    p.write_text(t)


shutil.copy2(HERE / "task_routing.py", L / "lingbotvla/models/vla/lingbot_vla/task_routing.py")
IMP = f"from lingbotvla.models.vla.lingbot_vla import task_routing as _tr  {MARK}\n"

# 1. MoE block: task routers next to the pretrained gate; router logits through task_routing
edit("lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py", [
    ("from logging import raiseExceptions\n", "from logging import raiseExceptions\n" + IMP),
    ("        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)\n",
     "        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)\n"
     f"        _tr.add_task_gates(self)  {MARK} task_gates.<t> (zeros here, copies of gate in the init checkpoint)\n"),
    ("            router_logits = F.linear(hidden_flat.float(), self.gate.weight.float())  # (B*T, num_experts)\n",
     f"            router_logits = _tr.router_logits(self, hidden_flat, batch_size, sequence_length)  {MARK} (B*T, E)\n"),
])

# 2. Qwen3-VL text layers: LoRA deltas on K and V in the joint-attention path
edit("lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py", [
    ("            key_state = self.self_attn.k_norm(self.self_attn.k_proj(hidden_states).view(hidden_shape))\n"
     "            value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)\n",
     f"            _k, _v = self.self_attn.k_proj(hidden_states), self.self_attn.v_proj(hidden_states)  {MARK}\n"
     "            _lora = getattr(self.self_attn, \"kv_lora\", None)\n"
     "            if _lora is not None:\n"
     "                _k = _k + _lora.delta_k(hidden_states)\n"
     "                _v = _v + _lora.delta_v(hidden_states)\n"
     "            key_state = self.self_attn.k_norm(_k.view(hidden_shape))\n"
     "            value_state = _v.view(hidden_shape)\n"),
])

# 3. Model: install LoRA, freeze rules, task ids from the batch
edit("lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py", [
    ("import einops\n", "import einops\n" + IMP),
    ("        self._install_moe_blocks()\n        self.pos_embeds = None\n",
     f"        self._install_moe_blocks()\n        _tr.install_kv_lora(self.qwenvl.model.language_model)  {MARK}\n"
     "        self.pos_embeds = None\n"),
    ("        if self.config.train_expert_only:\n"
     "            self.qwenvl.eval()\n"
     "            for params in self.qwenvl.parameters():\n"
     "                params.requires_grad = False\n\n"
     "    def train(self, mode: bool = True):\n",
     "        if self.config.train_expert_only:\n"
     "            self.qwenvl.eval()\n"
     "            for params in self.qwenvl.parameters():\n"
     "                params.requires_grad = False\n"
     f"        if _tr.enabled():  {MARK} LoRA always trains; the pretrained gate stays frozen next to task routers\n"
     "            for name, param in self.named_parameters():\n"
     "                if \".kv_lora.\" in name:\n"
     "                    param.requires_grad = True\n"
     "                elif name.endswith(\".mlp.gate.weight\") and _tr.config().get(\"task_routers\", 0):\n"
     "                    param.requires_grad = False\n\n"
     "    def train(self, mode: bool = True):\n"),
    ("        loss_dict = {}\n        if getattr(self.config, \"action_fp32\", False):\n",
     "        loss_dict = {}\n"
     f"        _task_id = kwargs.pop(\"task_id\", None)  {MARK}\n"
     "        if _tr.enabled():\n"
     "            if _task_id is None:\n"
     "                raise RuntimeError(\"task routing is on but the batch has no task_id\")\n"
     "            _tr.set_task_ids(_task_id.to(state.device))\n"
     "        if getattr(self.config, \"action_fp32\", False):\n"),
])

# 4. Data: task id per sample from the instruction (the dataloader workers inherit LINGBOT_ROUTE_CONFIG)
edit("lingbotvla/data/vla_data/utils.py", [
    ("        del batch_dict\n        batch_dict = {\n",
     f"        _route_prompts = batch_dict.get('prompt')  {MARK}\n        del batch_dict\n        batch_dict = {{\n"),
    ("        if image_grid_thw is not None:\n            batch_dict['image_grid_thw'] = image_grid_thw\n\n"
     "        if self.use_depth_align: \n",
     "        if image_grid_thw is not None:\n            batch_dict['image_grid_thw'] = image_grid_thw\n"
     f"        from lingbotvla.models.vla.lingbot_vla import task_routing as _tr  {MARK}\n"
     "        if _tr.enabled():\n"
     "            batch_dict['task_id'] = _tr.task_ids_from_prompts(_route_prompts)\n\n"
     "        if self.use_depth_align: \n"),
])

# 5. Deploy: route config from <ckpt>/route_config.json when the env var is unset (so the eval pool needs no change);
#    task id per observation from its instruction before the batched forward
edit("deploy/lingbot_vla_v2_policy.py", [
    ("        assert not (use_bf16 and use_fp32), 'Bfloat16 or Float32!!!'\n",
     "        assert not (use_bf16 and use_fp32), 'Bfloat16 or Float32!!!'\n"
     f"        _rc = os.path.join(str(path_to_pi_model), 'route_config.json')  {MARK}\n"
     "        if not os.environ.get('LINGBOT_ROUTE_CONFIG') and os.path.exists(_rc):\n"
     "            os.environ['LINGBOT_ROUTE_CONFIG'] = _rc\n"
     "            print(f'task routing config: {_rc}', flush=True)\n"),
    ("        applied = [self._prepare_model_input(obs) for obs in observations] # bsize, dict{key }\n",
     f"        from lingbotvla.models.vla.lingbot_vla import task_routing as _tr  {MARK}\n"
     "        if _tr.enabled():\n"
     "            _tr.set_from_prompts([str(o.get('prompt') or o.get('task') or o.get('instruction') or '') for o in observations])\n"
     "        applied = [self._prepare_model_input(obs) for obs in observations] # bsize, dict{key }\n"),
])
print("route patch applied to", L)
