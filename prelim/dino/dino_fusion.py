"""Frozen DINOv2 feature fusion for LingBot-VLA 2.0 (competition rule 2.2: public pretrained vision encoder).

DINOv2 (facebook/dinov2-with-registers-base, Apache-2.0, frozen) sees the same camera image as the Qwen3-VL vision
tower. Its patch features are pooled to the Qwen merger's token grid (8x8 per camera at img_size 256), normalized,
projected to the LLM width by a ZERO-initialized linear layer and added to the visual tokens in
FlowMatchingV2.embed_prefix. At initialization the model is therefore exactly the base model; sequence length,
attention masks, mrope positions and the deepstack path are unchanged. Only `norm` and `proj` are trained.

The image is reconstructed inside the model from the Qwen2-VL processor's pixel_values (patch 16, temporal 2,
merge 2, mean = std = 0.5), so neither the data pipeline nor the inference server needs changes, and DINOv2 sees
exactly what Qwen sees (training augmentations included). The DINOv2 weights live in the checkpoint
(model.dino_fusion.dino.*), so inference still loads one set of weights.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
QWEN_PATCH, QWEN_TEMPORAL, QWEN_MERGE = 16, 2, 2


def pixel_values_to_images(pixel_values: torch.Tensor) -> torch.Tensor:
    """Invert Qwen2VLImageProcessor patchification: (N, g*g, 3*2*16*16) -> (N, 3, g*16, g*16) in [0, 1]."""
    n, length, dim = pixel_values.shape
    g = int(round(length ** 0.5))
    assert g * g == length and g % QWEN_MERGE == 0 and dim == 3 * QWEN_TEMPORAL * QWEN_PATCH * QWEN_PATCH, (
        f"unexpected pixel_values layout {tuple(pixel_values.shape)}")
    m, p = QWEN_MERGE, QWEN_PATCH
    # rows: (hb, wb, mh, mw) with hb, wb over g // m merge blocks; columns: (c, t, ph, pw); both t slices are equal
    x = pixel_values.float().view(n, g // m, g // m, m, m, 3, QWEN_TEMPORAL, p, p)[:, :, :, :, :, :, 0]
    x = x.permute(0, 5, 1, 3, 6, 2, 4, 7).reshape(n, 3, g * p, g * p)
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


class DinoFusion(nn.Module):
    def __init__(self, dino_path: str, out_dim: int, input_size: int = 224, grid: int = 8):
        super().__init__()
        cfg = Dinov2WithRegistersConfig.from_pretrained(dino_path)
        cfg._attn_implementation = "sdpa"
        self.dino = Dinov2WithRegistersModel(cfg)  # weights come from the checkpoint (model.dino_fusion.dino.*)
        self.dino.requires_grad_(False)
        self.input_size, self.grid = input_size, grid
        self.num_prefix = 1 + cfg.num_register_tokens  # CLS + registers precede the patch tokens
        self.patch = cfg.patch_size
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.proj = nn.Linear(cfg.hidden_size, out_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()  # frozen: dropout / drop-path off (both are 0 for this checkpoint anyway)
        return self

    @torch.compiler.disable  # DINOv2's position-embedding interpolation is not compile-friendly; it is frozen anyway
    def dino_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = pixel_values_to_images(pixel_values)
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False,
                          antialias=True)
        mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        weight = self.dino.embeddings.patch_embeddings.projection.weight
        with torch.no_grad():
            h = self.dino(pixel_values=x.to(weight.dtype)).last_hidden_state[:, self.num_prefix:]
        side = self.input_size // self.patch
        h = h.transpose(1, 2).reshape(h.shape[0], -1, side, side)
        h = F.adaptive_avg_pool2d(h.float(), self.grid)  # -> Qwen merger grid, raster order like the visual tokens
        return h.flatten(2).transpose(1, 2)  # (N, grid*grid, hidden)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.dino_tokens(pixel_values)
        return self.proj(self.norm(tokens.to(self.norm.weight.dtype)))
