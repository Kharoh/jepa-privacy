"""Model definitions for MNIST experiments."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP


class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ViTLeJEPAEncoder(nn.Module):
    """ViT-based encoder + projection head for LeJEPA."""

    def __init__(
        self,
        proj_dim: int,
        img_size: int,
        in_chans: int = 1,
        backbone_name: str = "vit_tiny_patch16_224",
        patch_size: int = 16,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
            in_chans=in_chans,
            patch_size=patch_size,
        )
        emb_dim = self.backbone.num_features
        self.proj = MLP(emb_dim, [emb_dim * 4, emb_dim * 4, proj_dim], norm_layer=nn.BatchNorm1d)
        self.in_chans = in_chans
        self.img_size = img_size

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        b, v, _ = x.shape
        x = x.view(b * v, self.in_chans, self.img_size, self.img_size)
        emb = self.backbone(x)
        proj = self.proj(emb)
        emb = emb.view(b, v, -1)
        proj = proj.view(b, v, -1)
        return emb, proj


class LeJEPAEncoder(nn.Module):
    """Encoder with embedding and projection heads."""

    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        proj_dim: int,
        use_cnn: bool = False,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.use_cnn = use_cnn
        self.image_shape = image_shape
        self.input_dim = input_dim

        if use_cnn:
            channels, height, width = image_shape
            if input_dim != channels * height * width:
                raise ValueError("input_dim must match image_shape for CNN encoder")
            h2, w2 = height // 4, width // 4
            if height % 4 != 0 or width % 4 != 0:
                raise ValueError("image_shape height/width must be divisible by 4 for CNN encoder")
            self._cnn_feat_dim = 64 * h2 * w2
            self.backbone = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(self._cnn_feat_dim, emb_dim),
            )
        else:
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, emb_dim),
            )

        self.project = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        b, n, d = x.shape
        x_flat = x.reshape(b * n, d)

        if self.use_cnn:
            x_flat = x_flat.view(b * n, *self.image_shape)
            emb = self.backbone(x_flat).reshape(b, n, self.emb_dim)
        else:
            emb = self.backbone(x_flat).reshape(b, n, self.emb_dim)

        proj = self.project(emb.reshape(b * n, self.emb_dim)).reshape(b, n, self.proj_dim)

        return emb, proj


class LeJEPAModel(nn.Module):
    """
    LeJEPA: Learns invariant representations by minimizing variance
    across augmented views + SIGReg for Gaussian structure.
    """

    def __init__(
        self,
        input_dim: int,
        emb_dim: int = 64,
        proj_dim: int = 64,
        lamb: float = 0.5,
        use_cnn: bool = False,
        use_vit: bool = False,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
        vit_backbone: str = "vit_tiny_patch16_224",
    ):
        super().__init__()
        if use_vit:
            _, height, width = image_shape
            if height != width:
                raise ValueError("ViT backbone requires square images")
            self.encoder = ViTLeJEPAEncoder(
                proj_dim=proj_dim,
                img_size=height,
                in_chans=image_shape[0],
                backbone_name=vit_backbone,
            )
        else:
            self.encoder = LeJEPAEncoder(
                input_dim,
                emb_dim,
                proj_dim,
                use_cnn=use_cnn,
                image_shape=image_shape,
            )
        self.sigreg = SIGReg(knots=17)
        self.lamb = lamb

    def forward(self, x: torch.Tensor):
        return self.encoder(x)

    def compute_loss(self, x: torch.Tensor):
        """
        LeJEPA loss = λ * SIGReg(proj) + (1-λ) * inv_loss

        inv_loss: variance of projections across views
        """
        emb, proj = self.forward(x)

        proj_mean = proj.mean(dim=1, keepdim=True)
        inv_loss = (proj_mean - proj).square().mean()

        sigreg_loss = self.sigreg(proj.flatten(0, 1))

        total_loss = self.lamb * sigreg_loss + (1 - self.lamb) * inv_loss

        return {
            "total": total_loss,
            "inv": inv_loss,
            "sigreg": sigreg_loss,
            "emb": emb,
            "proj": proj,
        }


class MAEModel(nn.Module):
    """Masked Autoencoder with reconstruction objective."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        mask_ratio: float = 0.4,
        use_cnn: bool = False,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.mask_ratio = mask_ratio
        self.use_cnn = use_cnn
        self.image_shape = image_shape
        self.input_dim = input_dim

        if use_cnn:
            channels, height, width = image_shape
            if input_dim != channels * height * width:
                raise ValueError("input_dim must match image_shape for CNN MAE")
            if height % 4 != 0 or width % 4 != 0:
                raise ValueError("image_shape height/width must be divisible by 4 for CNN MAE")
            h2, w2 = height // 4, width // 4
            self._enc_hw = (h2, w2)
            self._cnn_feat_dim = 64 * h2 * w2

            self.encoder = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(self._cnn_feat_dim, latent_dim),
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, self._cnn_feat_dim),
                nn.ReLU(),
                nn.Unflatten(1, (64, h2, w2)),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, channels, kernel_size=2, stride=2),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim),
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim),
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[:, 0, :]
        if self.use_cnn:
            if x.dim() == 2:
                x = x.view(-1, *self.image_shape)
            return self.encoder(x)
        if x.dim() == 4:
            x = x.view(x.shape[0], -1)
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        """Returns reconstruction, latent, and mask."""
        if x.dim() == 3:
            x = x[:, 0, :]

        if x.dim() == 4:
            original = x.view(x.shape[0], -1)
        else:
            original = x

        mask = torch.rand_like(original) > self.mask_ratio
        masked_x = original * mask.float()

        if self.use_cnn:
            masked_img = masked_x.view(-1, *self.image_shape)
            latent = self.encoder(masked_img)
            recon_img = self.decoder(latent)
            recon = recon_img.view(recon_img.shape[0], -1)
        else:
            latent = self.encoder(masked_x)
            recon = self.decoder(latent)

        return recon, latent, mask, original

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        recon, _, mask, original = self.forward(x)
        return F.mse_loss(recon * (~mask).float(), original * (~mask).float())


class MAEViT(nn.Module):
    """ViT-based MAE for MNIST-sized images."""

    def __init__(
        self,
        img_size: int = 32,
        mask_ratio: float = 0.4,
        in_chans: int = 1,
        backbone_name: str = "vit_tiny_patch16_224",
        patch_size: int = 16,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
            in_chans=in_chans,
            patch_size=patch_size,
        )
        self.mask_ratio = mask_ratio
        patch_size = self.encoder.patch_embed.patch_size
        self.patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        self.num_patches = self.encoder.patch_embed.num_patches
        self.embed_dim = self.encoder.embed_dim
        self.in_chans = in_chans
        patch_dim = self.patch_size * self.patch_size * in_chans
        self.decoder = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, patch_dim),
        )

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        b, c, h, w = imgs.shape
        imgs = imgs.reshape(b, c, h // p, p, w // p, p)
        imgs = imgs.permute(0, 2, 4, 3, 5, 1)
        return imgs.reshape(b, -1, p * p * c)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        b, n, _ = patches.shape
        h = w = int((n) ** 0.5)
        patches = patches.reshape(b, h, w, p, p, self.in_chans)
        patches = patches.permute(0, 5, 1, 3, 2, 4)
        return patches.reshape(b, self.in_chans, h * p, w * p)

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.forward_features(imgs)
        if tokens.dim() == 2:
            return tokens
        if tokens.shape[1] == self.num_patches + 1:
            tokens = tokens[:, 1:, :]
        return tokens.mean(dim=1)

    def forward(self, imgs: torch.Tensor):
        if imgs.dim() == 3:
            imgs = imgs[:, 0, :]
        if imgs.dim() == 2:
            side = int(np.sqrt(imgs.shape[1]))
            imgs = imgs.view(-1, self.in_chans, side, side)
        patches = self.patchify(imgs)
        b, n, _ = patches.shape
        mask = torch.rand(b, n, device=imgs.device) < self.mask_ratio
        masked_patches = patches.clone()
        masked_patches[mask] = 0.0
        masked_imgs = self.unpatchify(masked_patches)
        tokens = self.encoder.forward_features(masked_imgs)
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(1).expand(b, n, -1)
        if tokens.shape[1] == self.num_patches + 1:
            tokens = tokens[:, 1:, :]
        pred = self.decoder(tokens)
        return pred, patches, mask

    def reconstruction_loss(self, imgs: torch.Tensor) -> torch.Tensor:
        pred, target, mask = self.forward(imgs)
        if mask.sum() == 0:
            return F.mse_loss(pred, target)
        return ((pred - target) ** 2)[mask].mean()
