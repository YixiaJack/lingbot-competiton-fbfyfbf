"""Patch: Qwen3-VL ViT attention falls back to sdpa on GPUs the installed flash-attn has no kernels for (sm_120)."""
from pathlib import Path
p = Path.home() / 'lingbot_train/code/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py'
s = p.read_text()
old = '''        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            if max_seqlen is None:'''
new = '''        attn_impl = _vit_attn_impl(self.config._attn_implementation)
        attention_interface: Callable = eager_attention_forward
        if attn_impl != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]

        if attn_impl == "flash_attention_2":
            if max_seqlen is None:'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
helper = '''

_VIT_FA2_OK = None


def _vit_attn_impl(impl):
    """flash-attn 2.8.3 in this image ships no sm_120 (RTX PRO 5000 Blackwell) kernels: use sdpa there."""
    global _VIT_FA2_OK
    if impl != "flash_attention_2":
        return impl
    if _VIT_FA2_OK is None:
        _VIT_FA2_OK = not (torch.cuda.is_available() and torch.cuda.get_device_capability() >= (12, 0))
    return impl if _VIT_FA2_OK else "sdpa"

'''
anchor = '\nclass '
i = s.index(anchor)
s = s[:i] + helper + s[i:]
p.write_text(s)
print('patched', p)
