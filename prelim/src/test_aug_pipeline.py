"""Self-checks for the training-time additions (run inside the lingbotvla env with the patched upstream importable):
task-uniform / motion-weighted sampling, wrist-camera dropout, Fourier amplitude modulation.

  PYTHONPATH=<patched lingbot-vla-v2>:<repo>/src python src/test_aug_pipeline.py <aug_config.json> <scratch_dir>
"""

import collections
import os
import sys
from pathlib import Path

import numpy as np
import torch

cfg, scratch = sys.argv[1], Path(sys.argv[2])
os.environ["LINGBOT_AUG_CONFIG"] = cfg
os.environ["LINGBOT_TASK_SAMPLING"] = "uniform"
os.environ["LINGBOT_FRAME_WEIGHTS"] = "1"
torch.manual_seed(0)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


# 1. sampling: three fake tasks with 100 / 400 / 1500 frames; the first 200 frames of task 1 have weight 0.3
import lingbot_sampling as S  # noqa: E402


class Sub:
    def __init__(self, n, root):
        self.num_frames, self.root = n, root


class VD:
    def __init__(self, n, root):
        self.dataset = Sub(n, root)


class Multi:
    pass


sizes = [100, 400, 1500]
multi = Multi()
multi._datasets, multi.dataset_start_index = [], [0, 100, 500]
for t, n in enumerate(sizes):
    root = scratch / f"t{t}"
    (root / "meta").mkdir(parents=True, exist_ok=True)
    w = np.ones(n, np.float32)
    if t == 1:
        w[:200] = 0.3
    np.save(root / "meta" / "frame_weights.npy", w)
    multi._datasets.append(VD(n, root))
out = np.array([S.remap_index(multi, i) for i in torch.randperm(2000).tolist() * 15])
share = [np.mean((out >= a) & (out < a + n)) for a, n in zip(multi.dataset_start_index, sizes)]
check("task shares uniform", all(abs(s - 1 / 3) < 0.02 for s in share), np.round(share, 3))
t1 = out[(out >= 100) & (out < 500)] - 100
check("low-motion frames down-weighted", abs(np.mean(t1 < 200) - 0.3 / 1.3) < 0.02, round(float(np.mean(t1 < 200)), 3))
check("indices in range", out.min() >= 0 and out.max() < 2000)

# 2 + 3. prepare_images with the patched transform.py
import lingbot_aug as A  # noqa: E402
import lingbotvla.data.vla_data.transform as T  # noqa: E402

check("patched transform uses lingbot_aug", T.apply_visual_augmentation.__module__ == "lingbot_aug")
keys = ["observation.images.camera_top", "observation.images.camera_wrist_left", "observation.images.camera_wrist_right"]
img = {k: torch.randint(0, 255, (3, 240, 320), dtype=torch.uint8) for k in keys}
obs = {"state": torch.zeros(14), "image": img}
patterns, n_fourier, n = collections.Counter(), collections.Counter(), 1500
for _ in range(n):
    ims, masks, pil, _, params = T.prepare_images(None, obs, image_keys=keys, train=True, use_depth_align=True,
                                                  return_augment_params=True)
    patterns[tuple(masks.tolist())] += 1
    assert masks[0], "head camera dropped"
    for i, m in enumerate(masks.tolist()):
        if not m:
            assert (ims[i] == -1).all() and (pil[i] == 0).all(), "dropped view not padded"
    fut, fmasks, fpil, _ = T.prepare_images(None, obs, image_keys=keys, train=True, use_depth_align=True,
                                           augment_params=params)
    assert fmasks.tolist() == masks.tolist(), "future frame dropout differs"
    if params.get("identity"):
        assert params.get("fourier") is None
    for slot in (params.get("fourier") or {}):
        n_fourier[slot] += 1
drop = 1 - patterns[(True, True, True)] / n
check("wrist dropout rate ~ p_wrist_drop", abs(drop - A.CFG["p_wrist_drop"]) < 0.04, f"{drop:.3f} vs {A.CFG['p_wrist_drop']}")
exp = A.CFG["p_fourier"] * A.CFG["p_fourier_cam"] * (1 - A.CFG["p_identity"])
rates = {k: round(v / n, 3) for k, v in n_fourier.items()}
check("fourier per-camera rate", all(abs(v - exp) < 0.05 for v in rates.values()) if exp else not rates, f"{rates} vs {exp:.3f}")
_, ev, _, _ = T.prepare_images(None, obs, image_keys=keys, train=False, use_depth_align=True)
check("no dropout at inference", ev.all())

# Fourier internals on a natural-ish image (smooth + edges)
x = torch.zeros(3, 240, 320)
x[:, 60:180, 80:240] = 0.8
x += 0.1 * torch.sin(torch.linspace(0, 20, 320))[None, None, :]
x = (x.clamp(0, 1) * 255).round().to(torch.uint8)
p = {"top": {"amp": 1.4, "strength": 0.6, "grid": 0.5 + torch.rand(8, 8)}}
y = A.apply_fourier(x, p, "observation.images.camera_top")
check("fourier keeps shape/dtype", y.shape == x.shape and y.dtype == x.dtype)
# DC gain is exactly 1; only clipping to [0, 255] can move the mean, so check DC on a mid-gray image that never clips
g = (96 + 0.25 * x.float()).round().to(torch.uint8)
gy = A.apply_fourier(g, p, "observation.images.camera_top")
check("no clipping on mid-gray test image", int(gy.min()) > 0 and int(gy.max()) < 255, f"range {int(gy.min())}..{int(gy.max())}")
check("fourier keeps mean (DC)", abs(gy.float().mean() - g.float().mean()) < 0.5, f"{g.float().mean():.2f} -> {gy.float().mean():.2f}")
fx, fy = torch.fft.fft2(x.float()), torch.fft.fft2(y.float())
strong = fx.abs() > fx.abs().flatten().quantile(0.99)
dphi = torch.angle(fy[strong] * fx[strong].conj()).abs()
check("fourier keeps phase", float(dphi.median()) < 0.05, f"median |dphase| {float(dphi.median()):.4f}")
check("fourier changes amplitude", float((y.float() - x.float()).abs().mean()) > 1.0)
check("camera without params untouched", torch.equal(A.apply_fourier(x, p, "camera_wrist_left"), x))
# teacher (pil) image must not see Fourier: same params with and without the fourier entry give the same pil
params = A.sample_visual_augmentation_params(None)
while params.get("identity"):
    params = A.sample_visual_augmentation_params(None)
params["wrist_drop"] = None
params["fourier"] = {s: {"amp": 1.5, "strength": 0.7, "grid": 0.45 + 1.1 * torch.rand(8, 8)} for s in A.CAMERA_SLOTS}
params["noise"], params["salt_pepper"] = 0.0, 0.0  # deterministic pixel path for the comparison
a_img, _, a_pil, _ = T.prepare_images(None, obs, image_keys=keys, train=True, use_depth_align=True, augment_params=params)
b_img, _, b_pil, _ = T.prepare_images(None, obs, image_keys=keys, train=True, use_depth_align=True,
                                      augment_params={**params, "fourier": None})
check("teacher image has no Fourier", torch.equal(a_pil, b_pil))
check("model image has Fourier", not torch.equal(a_img, b_img))
print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
