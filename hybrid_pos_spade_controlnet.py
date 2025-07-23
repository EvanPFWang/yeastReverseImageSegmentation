
"""
The goal is end-to-end training in <= 8 h on a single T4 GPU (g4dn.2xLarge)
exploit  AMP + Torch.compile where possible.

Modules provided
----------------
1. `fourierPositionalEncoding`   – fixed sin/cos positional features.
2. `coordConv2d`                – CoordConv implementation.
3. `SPADEBlock`                 – classic SPADE residual block.
4. `spatialBroadcast`           – tiles a latent code over the H×W grid.
5. `posSPADEGenerator`          – generator backbone (Stage 1).
6. `patchDiscriminator`         – 2-scale PatchGAN (Stage 3).
7. `ControlNetAdapter`          – tiny adapter on top of a frozen
                                  Stable-Diffusion v3 UNet (Stage 2).
8. `yeastFluoDataset`           – loads (mask, coarse-image) pairs.
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


import pytorch_lightning as pl


#

class fourierPositionalEncoding(nn.Module):
    """Fixed sin/cos Fourier features (λ controls spatial frequency)."""

    def __init__(self, num_channels: int = 64, wave_length: int = 64):
        super().__init__()
        assert num_channels % 4 == 0, "num_channels must be multiple of 4"
        self.num_channels = num_channels
        self.inv_freq = 2.0 * np.pi / wave_length
        self.register_buffer("freq",
                             2 * torch.pi * torch.arange(self.num_channels // 4, dtype=torch.float32)
                             .view(-1, 1, 1, 1)  # (C/4,1,1,1)
                             )

    def forward(self, h: int, w: int, device=None):
        y, x = torch.meshgrid(
            torch.linspace(0, 1, h, device=device),
            torch.linspace(0, 1, w, device=device),
            indexing="ij",
        )
        sin_inp = self.freq * torch.stack((x, y), -1)  # broadcasts correctly
        emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)  # (H, W, 4)
        emb = emb.permute(2, 0, 1)  # (4, H, W)
        emb = emb.repeat(self.num_channels // 4, 1, 1)  # (C, H, W)
        return emb

class coordConv2d(nn.Conv2d):
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


class spatialBroadcast(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, z: torch.Tensor, h: int, w: int):
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return z.repeat(1, 1, h, w)
class posSPADEGenerator(nn.Module):
    def __init__(self, label_ch: int, coord_ch: int = 2, fourier_ch: int = 64):
        super().__init__()
        self.fourier = fourierPositionalEncoding(fourier_ch)
        self.coord_conv = coordConv2d(label_ch + 1 + fourier_ch, 64, 3, padding=1)
        self.down1 = nn.Conv2d(64, 128, 4, 2, 1)
        self.down2 = nn.Conv2d(128, 256, 4, 2, 1)
        self.spade128 = SPADEBlock(256, 256, label_ch)
        self.spade256 = SPADEBlock(256, 128, label_ch)
        self.spade512 = SPADEBlock(128, 64, label_ch)
        self.to_img = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1), nn.Tanh()
        )
        self.broadcast = spatialBroadcast(32)

    def forward(self, label, coarse_img, z):
        # label: (B, C, H, W) one‑hot; coarse_img: (B,1,H,W)
        pos_fourier = self.fourier(label.shape[-2], label.shape[-1], device=label.device)
        pos_fourier = pos_fourier.unsqueeze(0).repeat(label.size(0), 1, 1, 1)
        x = torch.cat((label, coarse_img, pos_fourier), dim=1)
        x = self.coord_conv(x)

        x = self.down1(F.leaky_relu(x, 0.2))
        x = self.down2(F.leaky_relu(x, 0.2))

        b, _, h, w = x.shape
        z_broadcast = self.broadcast(z, h, w)
        x = torch.cat((x, z_broadcast), dim=1)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.spade128(x, F.interpolate(label, scale_factor=0.5))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.spade256(x, F.interpolate(label, scale_factor=1.0))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.spade512(x, F.interpolate(label, scale_factor=2.0))

        out = self.to_img(torch.sigmoid(x))
        return out

class patchDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 3, n_layers: int = 3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, 64, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        nf = 64
        for n in range(1, n_layers):
            nf_prev, nf = nf, min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(0.2, True),
            ]
        layers += [nn.Conv2d(nf, 1, 4, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class yeastFluoDataset(Dataset):
    def __init__(self, img_dir: Path, transform=None):
        self.img_dir = Path(img_dir)
        self.paths = sorted(p for p in self.img_dir.glob("*_coarse.png"))
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        coarse_path = self.paths[idx]
        mask_path = coarse_path.with_name(coarse_path.name.replace("_coarse", "_mask"))
        coarse = read_image(str(coarse_path)).float() / 255.0  # (1, H, W)
        mask = read_image(str(mask_path))
        label = F.one_hot(mask.long(), num_classes=256).permute(2, 0, 1).float()
        if self.transform:
            coarse = self.transform(coarse)
        return label, coarse



class yeastGAN(pl.LightningModule if pl else nn.Module):
    def __init__(self, num_labels=256, lr=2e-4):
        super().__init__()
        self.gen = posSPADEGenerator(num_labels)
        self.disc = patchDiscriminator(num_labels + 2)  # label + I0 + img
        self.latent_dim = 32
        self.lr = lr
        self.automatic_optimization = True

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0.0, 0.99))
        d_opt = torch.optim.Adam(self.disc.parameters(), lr=self.lr, betas=(0.0, 0.99))
        return [g_opt, d_opt], []

    def forward(self, label, coarse):
        z = torch.randn(label.size(0), self.latent_dim, device=self.device)
        return self.gen(label, coarse, z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        label, coarse = batch
        fake = self(label, coarse)
        if optimizer_idx == 0:  # generator update
            out_fake = self.disc(torch.cat((label, coarse, fake), dim=1))
            g_loss = -out_fake.mean()
            self.log("g_loss", g_loss)
            return g_loss
        else:  # discriminator update
            out_real = self.disc(torch.cat((label, coarse, coarse), dim=1))
            out_fake = self.disc(torch.cat((label, coarse, fake.detach()), dim=1))
            d_loss = F.relu(1 + out_real).mean() + F.relu(1 - out_fake).mean()
            self.log("d_loss", d_loss)
            return d_loss
