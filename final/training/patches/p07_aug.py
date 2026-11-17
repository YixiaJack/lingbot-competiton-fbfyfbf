"""p07: configurable colour-augmentation strength (lingbotvla/data/vla_data/transform.py, code_v3 worktree).

The official sampler draws brightness x[0.7,1.3], contrast x[0.6,1.4], saturation x[0.5,1.5] per sample (one draw shared
by all cameras and the future frame; no hue change, no crop/rotate). LB_AUG_BRIGHTNESS / LB_AUG_CONTRAST /
LB_AUG_SATURATION set the half-widths (defaults 0.3 / 0.4 / 0.5 = official). v3 uses 0.05 each: small on-site lighting
changes. Hue stays untouched (plate and fruit colours are the targets' identity). Enabled with data.image_augment=true.
"""
from pathlib import Path

p = Path.home() / 'lingbot_train/code_v3/lingbotvla/data/vla_data/transform.py'
s = p.read_text()
old = '''    _visual_hw(reference)
    device = reference.device
    return {
        "brightness": 0.7 + torch.rand((), device=device) * 0.6,
        "contrast": 0.6 + torch.rand((), device=device) * 0.8,
        "saturation": 0.5 + torch.rand((), device=device),
    }'''
new = '''    _visual_hw(reference)
    device = reference.device
    import os as _os
    b = float(_os.environ.get("LB_AUG_BRIGHTNESS", "0.3"))
    c = float(_os.environ.get("LB_AUG_CONTRAST", "0.4"))
    s = float(_os.environ.get("LB_AUG_SATURATION", "0.5"))
    return {
        "brightness": 1.0 - b + torch.rand((), device=device) * 2 * b,
        "contrast": 1.0 - c + torch.rand((), device=device) * 2 * c,
        "saturation": 1.0 - s + torch.rand((), device=device) * 2 * s,
    }'''
assert s.count(old) == 1, s.count(old)
p.write_text(s.replace(old, new))
print('patched p07', p)
