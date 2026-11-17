"""Whitelist-only training-time image augmentation for the LingBot-VLA 2.0 RoboTwin prelim.

Drop-in replacements for lingbotvla.data.vla_data.transform.{sample_visual_augmentation_params,
apply_visual_augmentation}; the patched transform.py switches to these when LINGBOT_AUG_CONFIG points at a JSON
config (configs/aug_whitelist.json). Inference never augments (upstream only calls these with train=True).

Every operation is on the competition's whitelist (rule 2.1.1: pixel-level degradation of the same clean scene):
  color jitter (brightness / contrast / saturation / color temperature), Gaussian + salt-and-pepper noise,
  Gaussian + slight motion blur, small random scale/translate (crop), random erasing (cutout), JPEG artifacts.
Nothing replaces backgrounds, inserts objects, relights, or changes camera parameters (rule 2.1.2 blacklist).
Optional wrist-camera dropout (p_wrist_drop) hides a wrist view from the model for that sample (training strategy,
rule 2.3; it synthesizes no pixels); the head camera is never dropped.
Optional Fourier amplitude modulation (p_fourier): whole-image FFT, the amplitude spectrum is multiplied by a smooth
random envelope while the phase is kept, i.e. a random zero-phase linear filter (a generalization of blur / sharpen);
per camera, reused on the future frame, applied to the model input only (never to the frozen teachers' images).
Optional random convolution (p_randconv, after "RandConv", Xu et al. ICLR 2021): one random k x k spatial kernel
(k from randconv_kernels) applied to the whole image, the SAME kernel on each of R, G, B (no cross-channel mixing), with
DC gain 1 (a box filter plus a zero-sum random part of scale randconv_std): flat regions keep their color, only local
texture / edges are scrambled -- colors named in instructions ("blue mat", "red block") stay true. The output is
re-standardized to the image's per-channel mean / std and blended with the original by a random weight. A random
linear filter of the same family as blur; no segmentation, no external images. One kernel per sample, replayed on every
view and frame (randomized scenes change the table / wall texture for all cameras at once).
A randconv sample skips the ops in randconv_excludes (default Fourier and blur, the same family of whole-image linear
filters); cutout, noise, salt-and-pepper and JPEG stay independent.

As upstream, one parameter set is sampled per training sample and replayed on every camera view and on the future
frames, so the current/future pair stays consistent for the video/depth alignment losses.
"""

import io
import json
import math
import os

import torch
import torch.nn.functional as F

DEFAULTS = {
    "p_identity": 0.3,           # fraction of samples left untouched (keeps clean-setting fidelity)
    "brightness": [0.6, 1.2],      # >1.2 washes the near-white clean scene out
    "contrast": [0.7, 1.3],
    "saturation": [0.5, 1.5],
    "color_temp": 0.15,          # R gain 1+t, B gain 1-t, t ~ U(-color_temp, color_temp)
    "p_geometric": 0.5,
    "scale": [0.92, 1.08],
    "translate": 0.06,           # fraction of width / height
    "p_blur": 0.3,
    "gauss_sigma": [0.3, 1.5],
    "p_motion": 0.3,             # share of blurred samples that use motion blur instead of Gaussian
    "motion_kernel": [3, 7],
    "p_noise": 0.3,
    "noise_sigma": [0.01, 0.05],
    "p_salt_pepper": 0.1,
    "salt_pepper_amount": [0.001, 0.006],
    "p_cutout": 0.5,
    "cutout_num": [1, 2],
    "cutout_area": [0.02, 0.12],
    "cutout_aspect": [0.5, 2.0],
    "p_jpeg": 0.3,
    "jpeg_quality": [30, 90],
    "p_wrist_drop": 0.0,         # per sample: hide wrist camera(s) from the model (head camera always kept)
    "wrist_drop_both": 0.5,      # when dropping: both wrists with this probability, else one at random
    # Fourier amplitude modulation of the whole image (PatchWAM-style), phase kept = a random zero-phase linear
    # filter (generalized blur / sharpen). Model input only: the teacher image is taken before it (patched
    # transform.py). Per sample with p_fourier, then every camera independently with p_fourier_cam.
    "p_fourier": 0.0,
    "p_fourier_cam": 0.5,
    "fourier_amp": [0.5, 1.5],         # global gain on all non-DC frequencies
    "fourier_env_strength": [0.0, 0.6],
    "fourier_env_grid": 8,             # 8x8 grid over the (centered) spectrum, bilinearly upsampled
    "fourier_env_range": [0.5, 1.5],
    "p_randconv": 0.0,                 # per sample (non-identity): random convolution
    "randconv_kernels": [3, 5],        # kernel size drawn from this list
    "randconv_std": 0.5,               # scale of the zero-sum random part of the kernel (texture scrambling strength)
    "randconv_mix": [0.0, 1.0],        # weight of the ORIGINAL image in the blend (0 = conv output only)
    "translate_y": None,               # vertical translate range if different from "translate" (table height)
    "randconv_excludes": ["fourier", "blur"],  # ops skipped on a randconv sample: same family (whole-image linear
                                       # filters), stacking them only deepens the distortion. Cutout (occlusion,
                                       # a different factor), noise, salt-and-pepper and JPEG stay independent.
}
CAMERA_SLOTS = ("top", "wrist_left", "wrist_right")


def _camera_slot(key):
    return next((s for s in CAMERA_SLOTS[1:] if s in key), "top")


def _load_cfg():
    cfg = dict(DEFAULTS)
    path = os.environ.get("LINGBOT_AUG_CONFIG")
    if path:
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


CFG = _load_cfg()


def _u(lo, hi):
    return float(lo + (hi - lo) * torch.rand(()))


def _p(prob):
    return bool(torch.rand(()) < prob)


def sample_visual_augmentation_params(reference=None) -> dict:
    """Sample one replayable augmentation config for all views/frames in a sample."""
    c = CFG
    # wrist-camera dropout is drawn independently of the pixel augmentation and applied in prepare_images (patched
    # transform.py): that view is replaced by the "missing camera" padding with mask False, current + future frames
    wrist = None
    if c["p_wrist_drop"] > 0 and _p(c["p_wrist_drop"]):
        wrist = ["left", "right"] if _p(c["wrist_drop_both"]) else [["left"], ["right"]][int(torch.randint(2, ()))]
    fourier = None
    if c["p_fourier"] > 0 and _p(c["p_fourier"]):
        g = c["fourier_env_grid"]
        fourier = {
            slot: {
                "amp": _u(*c["fourier_amp"]),
                "strength": _u(*c["fourier_env_strength"]),
                "grid": c["fourier_env_range"][0]
                + (c["fourier_env_range"][1] - c["fourier_env_range"][0]) * torch.rand(g, g),
            }
            for slot in CAMERA_SLOTS
            if _p(c["p_fourier_cam"])
        } or None
    if _p(c["p_identity"]):
        return {"identity": True, "wrist_drop": wrist, "fourier": None}  # identity samples stay pixel-exact
    params = {
        "identity": False,
        "brightness": _u(*c["brightness"]),
        "contrast": _u(*c["contrast"]),
        "saturation": _u(*c["saturation"]),
        "temp": _u(-c["color_temp"], c["color_temp"]),
        "geo": None, "blur": None, "noise": 0.0, "salt_pepper": 0.0, "cutout": [], "jpeg": 0, "wrist_drop": wrist,
        "fourier": fourier, "randconv": None,
    }
    if _p(c["p_geometric"]):
        t = c["translate"]
        ty = c["translate_y"] if c.get("translate_y") is not None else t
        params["geo"] = (_u(*c["scale"]), _u(-t, t), _u(-ty, ty))
    if c["p_randconv"] > 0 and _p(c["p_randconv"]):
        ks = c["randconv_kernels"]
        k = int(ks[int(torch.randint(len(ks), ()))])
        rnd = torch.randn(k, k) * c["randconv_std"]
        weight = rnd - rnd.mean() + 1.0 / (k * k)  # box filter (DC gain 1) + zero-sum random part; shared by R, G, B
        params["randconv"] = {"weight": weight, "mix": _u(*c["randconv_mix"])}
        if "fourier" in c["randconv_excludes"]:
            params["fourier"] = None
    if _p(c["p_blur"]) and not (params["randconv"] is not None and "blur" in c["randconv_excludes"]):
        if _p(c["p_motion"]):
            k = int(torch.randint(c["motion_kernel"][0], c["motion_kernel"][1] + 1, ()))
            params["blur"] = ("motion", k | 1, _u(0.0, math.pi))
        else:
            params["blur"] = ("gauss", _u(*c["gauss_sigma"]))
    if _p(c["p_noise"]):
        params["noise"] = _u(*c["noise_sigma"])
    if _p(c["p_salt_pepper"]):
        params["salt_pepper"] = _u(*c["salt_pepper_amount"])
    if _p(c["p_cutout"]):
        for _ in range(int(torch.randint(c["cutout_num"][0], c["cutout_num"][1] + 1, ()))):
            area, aspect = _u(*c["cutout_area"]), math.exp(_u(math.log(c["cutout_aspect"][0]), math.log(c["cutout_aspect"][1])))
            w, h = min(1.0, math.sqrt(area * aspect)), min(1.0, math.sqrt(area / aspect))
            params["cutout"].append((_u(0, 1 - w), _u(0, 1 - h), w, h, _u(0, 1)))
    if _p(c["p_jpeg"]):
        params["jpeg"] = int(torch.randint(c["jpeg_quality"][0], c["jpeg_quality"][1] + 1, ()))
    return params


def _geometric(x, scale, tx, ty):
    n = x.shape[0]
    theta = torch.tensor([[1.0 / scale, 0.0, -2.0 * tx], [0.0, 1.0 / scale, -2.0 * ty]], dtype=x.dtype)
    grid = F.affine_grid(theta.expand(n, 2, 3), list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _conv_per_channel(x, kernel):
    c = x.shape[1]
    kh, kw = kernel.shape
    weight = kernel.to(x.dtype).expand(c, 1, kh, kw)
    x = F.pad(x, (kw // 2, kw // 2, kh // 2, kh // 2), mode="replicate")
    return F.conv2d(x, weight, groups=c)


def _gauss_kernel(sigma):
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    return k[:, None] * k[None, :]


def _motion_kernel(size, angle):
    k = torch.zeros(size, size)
    c = size // 2
    for s in torch.linspace(-c, c, 4 * size):
        x, y = int(round(c + float(s) * math.cos(angle))), int(round(c + float(s) * math.sin(angle)))
        if 0 <= x < size and 0 <= y < size:
            k[y, x] = 1.0
    return k / k.sum()


def _randconv(x, weight, mix):
    """x (N,3,H,W) in [0,1]: the same random k x k kernel on every channel, re-standardized per image and channel to
    x's mean / std, blended with x."""
    y = _conv_per_channel(x, weight)
    mu_x, sd_x = x.mean(dim=(2, 3), keepdim=True), x.std(dim=(2, 3), keepdim=True)
    mu_y, sd_y = y.mean(dim=(2, 3), keepdim=True), y.std(dim=(2, 3), keepdim=True)
    y = (y - mu_y) / (sd_y + 1e-6) * sd_x + mu_x
    return (mix * x + (1.0 - mix) * y).clamp(0.0, 1.0)


def _jpeg(x_uint8, quality):
    from PIL import Image
    import numpy as np

    out = []
    for frame in x_uint8:  # (C,H,W)
        buf = io.BytesIO()
        Image.fromarray(frame.permute(1, 2, 0).numpy()).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out.append(torch.from_numpy(np.array(Image.open(buf).convert("RGB"))).permute(2, 0, 1))
    return torch.stack(out)


def apply_visual_augmentation(visual: torch.Tensor, params: dict) -> torch.Tensor:
    """Apply the sampled params to an image (C,H,W) or clip (T,C,H,W); keeps shape and dtype."""
    if params.get("identity", False):
        return visual
    squeeze = visual.ndim == 3
    x = visual.unsqueeze(0) if squeeze else visual
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if x.max() > 1.0:
        x = x / 255.0

    if params["geo"] is not None:
        x = _geometric(x, *params["geo"])

    # color jitter: brightness, contrast, saturation, color temperature
    x = x * params["brightness"]
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - mean) * params["contrast"] + mean
    gray = x.mean(dim=1, keepdim=True)
    x = gray + (x - gray) * params["saturation"]
    gains = torch.tensor([1.0 + params["temp"], 1.0, 1.0 - params["temp"]], dtype=x.dtype).view(1, 3, 1, 1)
    x = (x * gains).clamp(0.0, 1.0)

    if params.get("randconv") is not None:
        x = _randconv(x, params["randconv"]["weight"], params["randconv"]["mix"])

    if params["blur"] is not None:
        kernel = _gauss_kernel(params["blur"][1]) if params["blur"][0] == "gauss" else _motion_kernel(*params["blur"][1:])
        x = _conv_per_channel(x, kernel)
    if params["noise"] > 0:
        x = x + torch.randn_like(x) * params["noise"]
    if params["salt_pepper"] > 0:
        r = torch.rand_like(x[:, :1])
        x = torch.where(r < params["salt_pepper"] / 2, torch.zeros_like(x), x)
        x = torch.where(r > 1 - params["salt_pepper"] / 2, torch.ones_like(x), x)
    h, w = x.shape[-2:]
    for x0, y0, cw, ch, fill in params["cutout"]:
        x[..., int(y0 * h):int((y0 + ch) * h), int(x0 * w):int((x0 + cw) * w)] = fill
    x = x.clamp(0.0, 1.0)

    if params["jpeg"]:
        x = _jpeg((x * 255.0).round().to(torch.uint8), params["jpeg"]).to(torch.float32) / 255.0

    if orig_dtype == torch.uint8:
        x = (x * 255.0).round().to(torch.uint8)
    else:
        x = x.to(orig_dtype)
    return x.squeeze(0) if squeeze else x


def apply_fourier(visual: torch.Tensor, fourier: dict, key: str) -> torch.Tensor:
    """Whole-image FFT amplitude modulation for camera `key` (C,H,W) or (T,C,H,W); phase and DC are kept."""
    p = fourier.get(_camera_slot(key))
    if p is None:
        return visual
    squeeze = visual.ndim == 3
    x = visual.unsqueeze(0) if squeeze else visual
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    scale = 255.0 if x.max() > 1.0 else 1.0
    x = x / scale
    h, w = x.shape[-2:]
    grid = F.interpolate(p["grid"][None, None].float(), size=(h, w), mode="bilinear", align_corners=True)[0, 0]
    env = 1.0 + p["strength"] * (grid - 1.0)        # envelope over the centered spectrum
    gain = torch.fft.ifftshift(p["amp"] * env)       # back to FFT layout
    gain = 0.5 * (gain + torch.roll(torch.flip(gain, dims=(0, 1)), shifts=(1, 1), dims=(0, 1)))  # g(f) = g(-f)
    gain[0, 0] = 1.0                                 # keep the mean (DC)
    spec = torch.fft.fft2(x)                         # complex; real positive gain keeps the phase
    x = torch.fft.ifft2(spec * gain).real.clamp(0.0, 1.0) * scale
    x = x.round().to(orig_dtype) if orig_dtype == torch.uint8 else x.to(orig_dtype)
    return x.squeeze(0) if squeeze else x
