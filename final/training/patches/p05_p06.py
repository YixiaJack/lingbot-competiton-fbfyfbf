"""p05 (suggestion 3, RMSNorm / SwiGLU) and p06 (measurement) on the code_opt branch.

p05 LB_FUSED_OPS (default 0 until measured): the action expert's FixQwen2RMSNorm and the routed/shared SwiGLU MLPs
    call one shared torch.compile'd function each (fp32 reduction and the cast at the same place as the original),
    so the norm and the SiLU*up product run as single fused kernels even where the outer torch.compile graph breaks.
    The Q/K/V-prep Triton kernel of the suggestion is not ported: the outer torch.compile already fuses the cast, concat
    and rotary math between the decoder-unit calls; revisit only if the profile shows that region.
p06 LB_TIME_BREAKDOWN=1: synchronise around the teacher block so Depth_Forward_Time is real GPU time (it was host time
    of async launches). LB_PROFILE_STEPS=a,b: torch.profiler on rank 0 from step a to b, table to <output_dir>/profile.txt.
"""
from pathlib import Path

root = Path.home() / 'lingbot_train/code_opt'

# ------------------------------------------------------------------ p05
p = root / 'lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py'
s = p.read_text()
helpers = '''

import os as _lb_os


def _lb_rms_norm(x, weight, eps):
    dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps)).to(dtype)


def _lb_swiglu(gate, up):
    return torch.nn.functional.silu(gate) * up


_LB_FUSED = _lb_os.environ.get("LB_FUSED_OPS", "0") == "1"
if _LB_FUSED:
    _lb_rms_norm_c = torch.compile(_lb_rms_norm, dynamic=False)
    _lb_swiglu_c = torch.compile(_lb_swiglu, dynamic=False)

'''
anchor = '\nclass Qwen2MoeRoutedExpertMLP(nn.Module):'
assert s.count(anchor) == 1
s = s.replace(anchor, helpers + anchor)

old_mlp = '''    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))'''
new_mlp = '''    def forward(self, x):
        if _LB_FUSED and isinstance(self.act_fn, torch.nn.SiLU):
            return self.down_proj(_lb_swiglu_c(self.gate_proj(x), self.up_proj(x)))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))'''
n = s.count(old_mlp)
assert n == 2, ('mlp forwards', n)
s = s.replace(old_mlp, new_mlp)

old_norm = '''    def forward(self, hidden_states):
        # print(f'self.weight dtype is {self.weight.dtype}')
        input_dtype = hidden_states.dtype'''
new_norm = '''    def forward(self, hidden_states):
        if _LB_FUSED:
            return _lb_rms_norm_c(hidden_states, self.weight, self.variance_epsilon)
        # print(f'self.weight dtype is {self.weight.dtype}')
        input_dtype = hidden_states.dtype'''
assert s.count(old_norm) == 1, ('norm forward', s.count(old_norm))
s = s.replace(old_norm, new_norm)
p.write_text(s)
print('patched p05', p)

# ------------------------------------------------------------------ p06
p = root / 'tasks/vla/train_lingbotvla.py'
s = p.read_text()
old_t0 = '''                        depth_start_time = time.time()'''
new_t0 = '''                        if _LB_SYNC_T:
                            torch.cuda.synchronize()
                        depth_start_time = time.time()'''
assert s.count(old_t0) == 1
s = s.replace(old_t0, new_t0)
old_t1 = '''                        depth_forward_time = time.time() - depth_start_time'''
new_t1 = '''                        if _LB_SYNC_T:
                            torch.cuda.synchronize()
                        depth_forward_time = time.time() - depth_start_time'''
assert s.count(old_t1) == 1
s = s.replace(old_t1, new_t1)

old_step = '''            global_step += 1'''
new_step = '''            global_step += 1
            if _LB_PROF_RANGE and args.train.global_rank == 0:
                if global_step == _LB_PROF_RANGE[0]:
                    _lb_prof = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                    _lb_prof.start()
                elif global_step == _LB_PROF_RANGE[1] + 1 and _lb_prof is not None:
                    _lb_prof.stop()
                    with open(os.path.join(args.train.output_dir, "profile.txt"), "w") as _f:
                        _f.write(_lb_prof.key_averages().table(sort_by="cuda_time_total", row_limit=60))
                    _lb_prof = None'''
assert s.count(old_step) == 1
s = s.replace(old_step, new_step)

old_main = '''def main():
    args = parse_args(Arguments)'''
new_main = '''_LB_SYNC_T = os.environ.get("LB_TIME_BREAKDOWN", "0") == "1"
_LB_PROF_RANGE = tuple(int(x) for x in os.environ["LB_PROFILE_STEPS"].split(",")) if os.environ.get("LB_PROFILE_STEPS") else None
_lb_prof = None


def main():
    global _lb_prof
    args = parse_args(Arguments)'''
assert s.count(old_main) == 1
s = s.replace(old_main, new_main)
p.write_text(s)
print('patched p06', p)
