
"""
The goal is end-to-end training in <= 8 h on a single T4 GPU (g4dn.2xLarge)
exploit  AMP + Torch.compile where possible.

Modules provided
----------------
1. `FourierPositionalEncoding`   – fixed sin/cos positional features.
2. `CoordConv2d`                – CoordConv implementation.
3. `SPADEBlock`                 – classic SPADE residual block.
4. `SpatialBroadcast`           – tiles a latent code over the H×W grid.
5. `PosSPADEGenerator`          – generator backbone (Stage 1).
6. `PatchDiscriminator`         – 2-scale PatchGAN (Stage 3).
7. `ControlNetAdapter`          – tiny adapter on top of a frozen
                                  Stable-Diffusion v3 UNet (Stage 2).
8. `YeastFluoDataset`           – loads (mask, coarse-image) pairs.
9. `LightningModule`            – wraps the GAN losses & optimisers.

The **only mandatory external dependency** beyond the DLAMI python stack is
(https://github.com/huggingface/diffusers) (for the frozen
UNet and ControlNet utilities).  Install with:

```bash
pip install --upgrade diffusers accelerate transformers
```

Quick-start
===========
```bash
python hybrid_pos_spade_controlnet.py \
       --img_dir /path/to/2048_dataset/ \
       --real_dir /path/to/16_real_tiles/ \
       --epochs 2 --batch 4 --precision 16-mixed
```
The defaults mimic the 8-hour schedule: 30 k iters warm-up + 20 k iters
ControlNet + 5 k fine-tune.

"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image

# --- Positional encodings ----------------------------------------------------

class FourierPositionalEncoding(nn.Module):
    """Fixed sin/cos Fourier features (λ controls spatial frequency)."""

    def __init__(self, num_channels: int = 64, wave_length: int = 64):
        super().__init__()
        assert num_channels % 4 == 0, "num_channels must be multiple of 4"
        self.num_channels = num_channels
        self.inv_freq = 2.0 * np.pi / wave_length

    def forward(self, h: int, w: int, device=None):
        y, x = torch.meshgrid(
            torch.linspace(0, 1, h, device=device),
            torch.linspace(0, 1, w, device=device),
            indexing="ij",
        )
        sin_inp = self.inv_freq * torch.stack((x, y), dim=-1)  # (H, W, 2)
        emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)  # (H, W, 4)
        emb = emb.permute(2, 0, 1)  # (4, H, W)
        emb = emb.repeat(self.num_channels // 4, 1, 1)  # (C, H, W)
        return emb

class CoordConv2d(nn.Conv2d):
    """coordConv layer – appends normalised (x,y) to the input channels."""
    def forward(self, x: torch.Tensor):
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing="ij",
        )
        coords = torch.stack((xx, yy), dim=0).expand(b, -1, -1, -1)
        x = torch.cat((x, coords), dim=1)
        return super().forward(x)

class SPADEBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, label_ch: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_ch, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_ch, 128, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(128, in_ch, 3, padding=1)
        self.mlp_beta = nn.Conv2d(128, in_ch, 3, padding=1)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, seg: torch.Tensor):
        seg_feat = self.mlp_shared(seg)
        gamma = self.mlp_gamma(seg_feat)
        beta = self.mlp_beta(seg_feat)
        out = self.norm(x) * (1 + gamma) + beta
        out = self.act(out)
        out = self.conv(out)
        return out


class SpatialBroadcast(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, z: torch.Tensor, h: int, w: int):
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return z.repeat(1, 1, h, w)
    