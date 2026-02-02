"""
LeJEPA vs MAE comparison on ImageNette (inet10).

This script trains:
1) LeJEPA with SIGReg + invariance loss
2) MAE with a ViT-based encoder/decoder

It compares training losses and online linear probe accuracy.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import MLP
from torchvision.transforms import v2
import timm

IMAGENETTE_MEAN = [0.485, 0.456, 0.406]
IMAGENETTE_STD = [0.229, 0.224, 0.225]


@dataclass
class Config:
    img_size: int = 96
    batch_size: int = 64
    epochs: int = 20
    num_workers: int = 4
    lr: float = 2e-4
    probe_lr: float = 1e-3
    weight_decay: float = 5e-2
    lamb: float = 0.02
    num_views: int = 2
    proj_dim: int = 64
    mae_mask_ratio: float = 0.4
    seed: int = 42
    num_clients: int = 5
    samples_per_client: int = 1000
    dirichlet_alpha: float = 1.0
    num_rounds: int = 500
    local_epochs: int = 2
    eval_every: int = 50
    plot_rounds: int = 5
    dp_enabled: bool = False
    dp_clip_norm: float = 1.0
    dp_noise_multiplier: float = 0.8


@dataclass
class DPConfig:
    enabled: bool = False
    clip_norm: float = 1.0
    noise_multiplier: float = 0.8
    seed: int = 42
    apply_to_gradients: bool = True
    apply_to_updates: bool = True


def _disable_cuda_sdp_kernels() -> None:
    if not torch.cuda.is_available():
        return
    backend = torch.backends.cuda
    # Prefer math-only SDP kernels to avoid missing backward implementations.
    if hasattr(backend, "sdp_kernel"):
        backend.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False)
    for attr in (
        "sdp_enabled",
        "flash_sdp_enabled",
        "scaled_dot_product_efficient_attention_enabled",
        "enable_flash_sdp",
        "enable_mem_efficient_sdp",
        "enable_math_sdp",
    ):
        if hasattr(backend, attr):
            try:
                if "math" in attr:
                    getattr(backend, attr)(True)
                else:
                    getattr(backend, attr)(False)
            except TypeError:
                # Fallback for boolean properties on older versions.
                setattr(backend, attr, False)


def _global_l2_norm(tensors: List[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        return torch.tensor(0.0)
    device = tensors[0].device
    total = torch.zeros((), device=device)
    for t in tensors:
        total = total + t.pow(2).sum()
    return total.sqrt()


def _clip_tensors(tensors: List[torch.Tensor], clip_norm: float) -> Tuple[List[torch.Tensor], float]:
    if not tensors:
        return [], 0.0
    norm = _global_l2_norm(tensors)
    norm_value = norm.item()
    if clip_norm <= 0:
        return tensors, norm_value
    scale = min(1.0, clip_norm / (norm_value + 1e-12))
    return [t * scale for t in tensors], norm_value


def _add_noise(tensors: List[torch.Tensor], std: float, generator: torch.Generator) -> List[torch.Tensor]:
    if std <= 0:
        return tensors
    return [t + torch.randn(t.shape, device=t.device, dtype=t.dtype, generator=generator) * std for t in tensors]


def apply_dp_to_tensors(
    tensors: List[torch.Tensor],
    dp_config: DPConfig,
    generator: torch.Generator,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    if not dp_config.enabled:
        return tensors, {"norm": _global_l2_norm(tensors).item() if tensors else 0.0}

    clipped, norm_value = _clip_tensors(tensors, dp_config.clip_norm)
    noise_std = dp_config.noise_multiplier * dp_config.clip_norm
    noised = _add_noise(clipped, noise_std, generator)
    return noised, {"norm": norm_value, "noise_std": noise_std}


def apply_dp_to_vector(
    vector: torch.Tensor,
    dp_config: DPConfig,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    tensors, stats = apply_dp_to_tensors([vector], dp_config, generator)
    return tensors[0], stats


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        a = torch.randn(proj.size(-1), 256, device=proj.device)
        a = a.div_(a.norm(p=2, dim=0))
        x_t = (proj @ a).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ViTEncoder(nn.Module):
    def __init__(self, proj_dim: int, img_size: int = 128):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            img_size=img_size,
        )
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 6 and x.shape[2] == 1:
            x = x.squeeze(2)

        if x.dim() == 4:
            x = x.unsqueeze(1)
        elif x.dim() == 5:
            if x.shape[0] < x.shape[1]:
                x = x.permute(1, 0, 2, 3, 4)
        else:
            raise ValueError(f"Expected 4D/5D input, got shape={tuple(x.shape)}")

        n, v = x.shape[:2]
        emb = self.backbone(x.reshape(n * v, *x.shape[2:]))
        proj = self.proj(emb).reshape(n, v, -1).transpose(0, 1)
        return emb, proj


class LeJEPAModel(nn.Module):
    def __init__(self, proj_dim: int, img_size: int = 128, lamb: float = 0.02):
        super().__init__()
        self.encoder = ViTEncoder(proj_dim=proj_dim, img_size=img_size)
        self.sigreg = SIGReg()
        self.lamb = lamb

    def compute_loss(self, views: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb, proj = self.encoder(views)
        inv_loss = (proj.mean(0) - proj).square().mean()
        sigreg_loss = self.sigreg(proj)
        lejepa_loss = sigreg_loss * self.lamb + inv_loss * (1 - self.lamb)
        return {
            "total": lejepa_loss,
            "inv": inv_loss,
            "sigreg": sigreg_loss,
            "emb": emb,
        }


class MAEViT(nn.Module):
    def __init__(self, img_size: int = 128, mask_ratio: float = 0.4):
        super().__init__()
        self.encoder = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=0,
            drop_path_rate=0.1,
            img_size=img_size,
        )
        self.mask_ratio = mask_ratio
        patch_size = self.encoder.patch_embed.patch_size
        self.patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        self.num_patches = self.encoder.patch_embed.num_patches
        self.embed_dim = self.encoder.embed_dim
        patch_dim = self.patch_size * self.patch_size * 3
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
        patches = patches.reshape(b, h, w, p, p, 3)
        patches = patches.permute(0, 5, 1, 3, 2, 4)
        return patches.reshape(b, 3, h * p, w * p)

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.forward_features(imgs)
        if tokens.dim() == 2:
            return tokens
        if tokens.shape[1] == self.num_patches + 1:
            tokens = tokens[:, 1:, :]
        return tokens.mean(dim=1)

    def forward(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            return torch.tensor(0.0, device=imgs.device)
        return ((pred - target) ** 2)[mask].mean()


class ImageNetteDataset(Dataset):
    def __init__(self, split: str, views: int, img_size: int):
        self.views = views
        self.ds = load_dataset("frgfm/imagenette", "160px", split=split)
        self.train_transform = v2.Compose(
            [
                v2.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
                v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.test_transform = v2.Compose(
            [
                v2.Resize(img_size),
                v2.CenterCrop(img_size),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.ds[idx]
        img = item["image"].convert("RGB")
        transform = self.train_transform if self.views > 1 else self.test_transform
        views = torch.stack([transform(img) for _ in range(self.views)])
        return views, int(item["label"])


class ViewAugmenter:
    """Create multiple augmented views of the same batch for ImageNette."""
    def __init__(self, num_views: int, img_size: int):
        self.num_views = num_views
        self.transform = v2.Compose(
            [
                v2.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
                v2.RandomApply([v2.RandomSolarize(threshold=0.5)], p=0.2),
                v2.RandomHorizontalFlip(),
            ]
        )
        self.normalize = v2.Normalize(mean=IMAGENETTE_MEAN, std=IMAGENETTE_STD)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("Expected batch of images with shape (B, C, H, W)")
        views = []
        for _ in range(self.num_views):
            batch_views = []
            for img in x:
                img = _unnormalize(img.detach().cpu()).clamp(0, 1)
                aug = self.transform(img)
                batch_views.append(self.normalize(aug))
            batch_views = torch.stack(batch_views)
            views.append(batch_views)
        return torch.stack(views, dim=1)


def create_imagenette_transform(img_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize(img_size),
            v2.CenterCrop(img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENETTE_MEAN, std=IMAGENETTE_STD),
        ]
    )


def load_imagenette_non_iid(
    num_clients: int,
    total_samples: int,
    alpha: float,
    seed: int,
    img_size: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    rng = np.random.default_rng(seed)
    dataset = load_dataset("frgfm/imagenette", "160px", split="train")
    transform = create_imagenette_transform(img_size)

    total_samples = min(total_samples, len(dataset))
    subset_indices = rng.permutation(len(dataset))[:total_samples]

    class_indices = {c: [] for c in range(10)}
    for idx in subset_indices:
        class_indices[int(dataset[int(idx)]["label"])].append(int(idx))

    for c in class_indices:
        rng.shuffle(class_indices[c])

    client_indices = [[] for _ in range(num_clients)]
    for c in range(10):
        idxs = class_indices[c]
        if len(idxs) == 0:
            continue
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        raw_counts = proportions * len(idxs)
        counts = np.floor(raw_counts).astype(int)
        remainder = len(idxs) - counts.sum()
        if remainder > 0:
            fractional = raw_counts - counts
            for i in np.argsort(-fractional)[:remainder]:
                counts[i] += 1

        cursor = 0
        for client_id, count in enumerate(counts):
            if count == 0:
                continue
            client_indices[client_id].extend(idxs[cursor:cursor + count])
            cursor += count

    client_data = []
    client_labels = []
    for idxs in client_indices:
        if len(idxs) == 0:
            fallback_idx = int(rng.integers(0, len(dataset)))
            idxs = [fallback_idx]
        images = []
        labels = []
        for idx in idxs:
            item = dataset[int(idx)]
            img = item["image"].convert("RGB")
            images.append(transform(img))
            labels.append(int(item["label"]))
        client_data.append(torch.stack(images))
        client_labels.append(torch.tensor(labels, dtype=torch.long))

    return client_data, client_labels


def sample_tensor_dataset(
    data: torch.Tensor,
    labels: torch.Tensor,
    max_samples: int,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= max_samples:
        return data, labels
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data))[:max_samples]
    return data[indices], labels[indices]


def plot_client_class_distribution(
    client_labels: List[torch.Tensor],
    save_path: str = "client_class_distribution.png",
) -> np.ndarray:
    counts = np.zeros((len(client_labels), 10), dtype=int)
    for client_id, labels in enumerate(client_labels):
        unique, freqs = np.unique(labels.numpy(), return_counts=True)
        counts[client_id, unique] = freqs

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(counts, aspect="auto", cmap="viridis")
    ax.set_title("ImageNette class distribution per client")
    ax.set_xlabel("Class")
    ax.set_ylabel("Client")
    ax.set_xticks(list(range(10)))
    ax.set_yticks(list(range(len(client_labels))))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Samples")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return counts


def _unnormalize(imgs: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENETTE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENETTE_STD).view(1, 3, 1, 1)
    return imgs * std + mean


def plot_reconstructions(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    save_path: str,
    title: str,
    num_images: int = 4,
) -> None:
    # Gradient inversion can return tensors shaped like (B, V, 1, C, H, W).
    # Normalize to (B, C, H, W) by removing singleton dimensions and picking view 0.
    if original.dim() == 6 and original.shape[2] == 1:
        original = original.squeeze(2)
    if reconstructed.dim() == 6 and reconstructed.shape[2] == 1:
        reconstructed = reconstructed.squeeze(2)

    if original.dim() == 6 and original.shape[1] == 1:
        original = original.squeeze(1)
    if reconstructed.dim() == 6 and reconstructed.shape[1] == 1:
        reconstructed = reconstructed.squeeze(1)

    if original.dim() == 5:
        original = original[:, 0, :, :, :]
    if reconstructed.dim() == 5:
        reconstructed = reconstructed[:, 0, :, :, :]

    if original.dim() == 5:
        original = original.squeeze(1)
    if reconstructed.dim() == 5:
        reconstructed = reconstructed.squeeze(1)

    if original.dim() != 4 or reconstructed.dim() != 4:
        raise ValueError(
            "Expected 4D tensors after squeezing views; got "
            f"original={tuple(original.shape)}, reconstructed={tuple(reconstructed.shape)}"
        )

    original = _unnormalize(original[:num_images].cpu()).clamp(0, 1)
    reconstructed = _unnormalize(reconstructed[:num_images].cpu()).clamp(0, 1)

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.4, 3.2))
    for i in range(num_images):
        axes[0, i].imshow(original[i].permute(1, 2, 0))
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed[i].permute(1, 2, 0))
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


class GradientInversionAttack:
    """Reconstruct data from gradients to measure privacy leakage."""
    def __init__(self, model: nn.Module):
        self.model = model

    def attack(
        self,
        original_data: torch.Tensor,
        true_grad: torch.Tensor,
        iterations: int = 300,
        lr: float = 0.05,
        return_history: bool = False,
        record_steps: List[int] | None = None,
        loss_strategy: str = "cosine",
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        device = next(self.model.parameters()).device
        original_data = original_data.to(device)
        true_grad = true_grad.to(device)

        dummy = (torch.randn_like(original_data) * 0.1).requires_grad_(True)
        opt = torch.optim.Adam([dummy], lr=lr)
        tv_weight = 1e-4

        true_grad = true_grad.detach()
        history = {}
        record_steps = set(record_steps or [])
        if 0 in record_steps:
            history[0] = dummy.detach().cpu().clone()

        for step in range(1, iterations + 1):
            opt.zero_grad()
            self.model.zero_grad(set_to_none=True)

            if isinstance(self.model, LeJEPAModel):
                loss = self.model.compute_loss(dummy)["total"]
            elif isinstance(self.model, MAEViT):
                loss = self.model.reconstruction_loss(dummy)
            else:
                raise ValueError("Unsupported model type for gradient inversion")

            param_grads = torch.autograd.grad(
                loss,
                [p for p in self.model.parameters() if p.requires_grad],
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )

            grads_flat = [g.flatten() for g in param_grads if g is not None]
            if len(grads_flat) == 0:
                break

            dummy_grad = torch.cat(grads_flat)

            if loss_strategy == "cosine":
                grad_loss = 1.0 - F.cosine_similarity(
                    dummy_grad.unsqueeze(0), true_grad.unsqueeze(0)
                )
            elif loss_strategy == "mse":
                grad_loss = F.mse_loss(dummy_grad, true_grad)
            else:
                raise ValueError(f"Unknown loss_strategy: {loss_strategy}")

            if dummy.dim() >= 4:
                dummy_img = dummy.view(-1, *dummy.shape[-3:])
                h_diff = dummy_img[:, :, 1:, :] - dummy_img[:, :, :-1, :]
                w_diff = dummy_img[:, :, :, 1:] - dummy_img[:, :, :, :-1]
                tv_loss = torch.sum(torch.abs(h_diff)) + torch.sum(torch.abs(w_diff))
            else:
                tv_loss = torch.tensor(0.0, device=device)

            total_loss = grad_loss + (tv_weight * tv_loss)
            total_loss.backward()
            opt.step()

            if step in record_steps:
                history[step] = dummy.detach().cpu().clone()

        if return_history:
            return dummy.detach().cpu(), history
        return dummy.detach().cpu(), {}

    def compute_metrics(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        if original.dim() == 6 and original.shape[2] == 1:
            original = original.squeeze(2)
        if reconstructed.dim() == 6 and reconstructed.shape[2] == 1:
            reconstructed = reconstructed.squeeze(2)

        if original.dim() == 6 and original.shape[1] == 1:
            original = original.squeeze(1)
        if reconstructed.dim() == 6 and reconstructed.shape[1] == 1:
            reconstructed = reconstructed.squeeze(1)

        if original.dim() == 5:
            original = original[:, 0]
        if reconstructed.dim() == 5:
            reconstructed = reconstructed[:, 0]

        metrics: Dict[str, float] = {}
        metrics["mse"] = F.mse_loss(reconstructed, original).item()
        mse = metrics["mse"]
        metrics["psnr"] = 10 * np.log10(1.0 / (mse + 1e-10))

        orig_flat = original.flatten()
        recon_flat = reconstructed.flatten()
        metrics["cosine_sim"] = F.cosine_similarity(
            orig_flat.unsqueeze(0), recon_flat.unsqueeze(0)
        ).item()
        metrics["rel_l2"] = torch.norm(original - reconstructed).item() / (
            torch.norm(original).item() + 1e-10
        )

        return metrics


class FederatedClient:
    def __init__(
        self,
        client_id: int,
        data: torch.Tensor,
        model_type: str,
        num_views: int,
        device: torch.device,
        dp_config: DPConfig,
        img_size: int,
    ):
        self.client_id = client_id
        self.data = data
        self.model_type = model_type
        self.num_views = num_views
        self.device = device
        self.dp_config = dp_config
        gen_device = "cuda" if device.type == "cuda" else "cpu"
        self.dp_generator = torch.Generator(device=gen_device)
        self.dp_generator.manual_seed(self.dp_config.seed + client_id)
        self.augmenter = ViewAugmenter(num_views=num_views, img_size=img_size)

    def local_train(self, global_model: nn.Module, epochs: int, lr: float) -> Dict[str, torch.Tensor]:
        local_model = copy.deepcopy(global_model).to(self.device)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        use_amp = self.device.type == "cuda"
        scaler = GradScaler(enabled=use_amp)

        for _ in range(epochs):
            batch_size = min(16, len(self.data))
            indices = torch.randperm(len(self.data))[:batch_size]
            x_batch = self.data[indices]

            optimizer.zero_grad()
            if self.model_type == "lejepa":
                x_views = self.augmenter(x_batch)
                x_views = x_views.to(self.device)
                with autocast(device_type=self.device.type, enabled=use_amp):
                    loss_dict = local_model.compute_loss(x_views)
                    loss = loss_dict["total"]
                    inv_loss = loss_dict["inv"].item()
                    sigreg_loss = loss_dict["sigreg"].item()
            else:
                imgs = x_batch.to(self.device)
                with autocast(device_type=self.device.type, enabled=use_amp):
                    loss = local_model.reconstruction_loss(imgs)
                inv_loss = None
                sigreg_loss = None

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            test_indices = torch.randperm(len(self.data))[:16]
            x_test = self.data[test_indices]

        if self.model_type == "lejepa":
            x_test_views = self.augmenter(x_test).to(self.device)
            local_model.zero_grad()
            with autocast(device_type=self.device.type, enabled=use_amp):
                loss_dict = local_model.compute_loss(x_test_views)
                loss = loss_dict["total"]
                inv_loss = loss_dict["inv"].item()
                sigreg_loss = loss_dict["sigreg"].item()
        else:
            local_model.zero_grad()
            with autocast(device_type=self.device.type, enabled=use_amp):
                loss = local_model.reconstruction_loss(x_test.to(self.device))

        loss.backward()
        flat_grad = torch.cat(
            [p.grad.flatten() for p in local_model.parameters() if p.grad is not None]
        ).detach()

        if self.dp_config.enabled and self.dp_config.apply_to_gradients:
            flat_grad, _ = apply_dp_to_vector(flat_grad, self.dp_config, self.dp_generator)

        flat_grad = flat_grad.cpu()

        if self.model_type == "lejepa":
            x_flat = x_test_views.detach().cpu()
        else:
            x_flat = x_test

        if self.dp_config.enabled and self.dp_config.apply_to_updates:
            with torch.no_grad():
                global_state = global_model.state_dict()
                local_state = local_model.state_dict()
                delta_state = {k: (local_state[k] - global_state[k]).detach() for k in global_state}
                delta_values = [delta_state[k] for k in delta_state]
                dp_delta_values, _ = apply_dp_to_tensors(delta_values, self.dp_config, self.dp_generator)
                dp_delta_state = {k: dp_delta_values[i] for i, k in enumerate(delta_state.keys())}
                dp_state = {k: (global_state[k] + dp_delta_state[k]) for k in global_state}
        else:
            dp_state = local_model.state_dict()

        return {
            "gradients": flat_grad,
            "data": x_flat.detach(),
            "loss": loss.item(),
            "inv_loss": inv_loss,
            "sigreg_loss": sigreg_loss,
            "model_state": dp_state,
        }


class FederatedServer:
    def __init__(self, model: nn.Module):
        self.global_model = model

    def aggregate(self, client_updates: List[Dict[str, torch.Tensor]]) -> None:
        with torch.no_grad():
            avg_state = self.global_model.state_dict()
            device = next(self.global_model.parameters()).device
            for key, base_tensor in avg_state.items():
                client_tensors = [client["model_state"][key].to(device) for client in client_updates]
                if base_tensor.is_floating_point() or torch.is_complex(base_tensor):
                    stacked = torch.stack([t.to(base_tensor.dtype) for t in client_tensors], dim=0)
                    avg_state[key] = stacked.mean(dim=0)
                else:
                    # Preserve non-float parameters (e.g., buffers) from the first client
                    avg_state[key] = client_tensors[0]
            self.global_model.load_state_dict(avg_state)


def sample_tensor_batch(data: torch.Tensor, max_samples: int, seed: int = 42) -> torch.Tensor:
    if len(data) <= max_samples:
        return data
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data))[:max_samples]
    return data[indices]


def compute_gradients_for_data(
    model: nn.Module,
    x: torch.Tensor,
    model_type: str,
    augmenter: ViewAugmenter,
) -> torch.Tensor:
    device = next(model.parameters()).device
    model.zero_grad(set_to_none=True)

    if model_type == "lejepa":
        x_views = augmenter(x).to(device)
        loss = model.compute_loss(x_views)["total"]
    else:
        loss = model.reconstruction_loss(x.to(device))

    loss.backward()
    grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    return grads.detach().cpu()


def plot_metric_curve(x: List[int], y_a: List[float], y_b: List[float], label_a: str,
                      label_b: str, ylabel: str, title: str, save_path: str) -> None:
    if not x:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y_a, marker="o", label=label_a)
    ax.plot(x, y_b, marker="o", label=label_b)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def train_linear_probe(features: torch.Tensor, labels: torch.Tensor,
                        test_features: torch.Tensor, test_labels: torch.Tensor,
                        epochs: int = 10, lr: float = 1e-2) -> float:
    device = features.device
    probe = nn.Sequential(
        nn.BatchNorm1d(features.shape[1], affine=False),
        nn.Linear(features.shape[1], 10),
    ).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    probe.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = probe(features)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(test_features).argmax(dim=1)
        acc = (preds == test_labels).float().mean().item()
    return acc


def extract_features(model: nn.Module, loader: DataLoader, model_type: str,
                     device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for views, y in loader:
            views = views.to(device)
            if model_type == "lejepa":
                emb = model.encoder(views)[0]
                feat = emb.mean(dim=0)
            else:
                feat = model.encode(views[:, 0])
            feats.append(feat)
            labels.append(y)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def extract_features_from_tensors(
    model: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    model_type: str,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    feats = []
    lbls = []
    for idx in range(0, len(data), batch_size):
        batch = data[idx:idx + batch_size].to(device)
        if model_type == "lejepa":
            views = batch.unsqueeze(1)
            emb = model.encoder(views)[0]
            feat = emb
        else:
            feat = model.encode(batch)
        feats.append(feat.detach().cpu())
        lbls.append(labels[idx:idx + batch_size].cpu())
    return torch.cat(feats, dim=0), torch.cat(lbls, dim=0)


def train_linear_probe_from_tensors(
    model: nn.Module,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    model_type: str,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-2,
    batch_size: int = 256,
) -> float:
    train_features, train_labels = extract_features_from_tensors(
        model, train_data, train_labels, model_type, batch_size, device
    )
    test_features, test_labels = extract_features_from_tensors(
        model, test_data, test_labels, model_type, batch_size, device
    )

    train_features = train_features.to(device)
    test_features = test_features.to(device)
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)

    probe = nn.Sequential(
        nn.BatchNorm1d(train_features.shape[1], affine=False),
        nn.Linear(train_features.shape[1], 10),
    ).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    probe.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = probe(train_features)
        loss = criterion(logits, train_labels)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(test_features).argmax(dim=1)
        acc = (preds == test_labels).float().mean().item()
    return acc


def initialize_loss_log(log_path: str) -> None:
    with open(log_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "round",
            "model",
            "scope",
            "client_id",
            "loss_total",
            "loss_inv",
            "loss_sigreg",
        ])


def append_loss_log(log_path: str, round_idx: int, model: str, scope: str,
                    client_id: int, loss_components: Dict[str, float]) -> None:
    with open(log_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            round_idx,
            model,
            scope,
            client_id,
            loss_components.get("total"),
            loss_components.get("inv"),
            loss_components.get("sigreg"),
        ])


def run_federated_privacy_experiment(cfg: Config) -> Dict[str, Dict[str, List[float]]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    _disable_cuda_sdp_kernels()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_samples = cfg.num_clients * cfg.samples_per_client
    client_data, client_labels = load_imagenette_non_iid(
        num_clients=cfg.num_clients,
        total_samples=total_samples,
        alpha=cfg.dirichlet_alpha,
        seed=cfg.seed,
        img_size=cfg.img_size,
    )
    all_train_data = torch.cat(client_data)
    all_train_labels = torch.cat(client_labels)
    class_counts = plot_client_class_distribution(client_labels)
    for c in range(cfg.num_clients):
        print(f"  Client {c}: samples={len(client_data[c])}, class_counts={class_counts[c].tolist()}")

    val_dataset = load_dataset("frgfm/imagenette", "160px", split="validation")
    val_transform = create_imagenette_transform(cfg.img_size)
    val_images = []
    val_labels = []
    for idx in range(min(2000, len(val_dataset))):
        item = val_dataset[int(idx)]
        val_images.append(val_transform(item["image"].convert("RGB")))
        val_labels.append(int(item["label"]))
    val_data = torch.stack(val_images)
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    lejepa_model = LeJEPAModel(proj_dim=cfg.proj_dim, img_size=cfg.img_size, lamb=cfg.lamb).to(device)
    mae_model = MAEViT(img_size=cfg.img_size, mask_ratio=cfg.mae_mask_ratio).to(device)

    dp_config = DPConfig(
        enabled=cfg.dp_enabled,
        clip_norm=cfg.dp_clip_norm,
        noise_multiplier=cfg.dp_noise_multiplier,
        seed=cfg.seed,
        apply_to_gradients=True,
        apply_to_updates=True,
    )

    lejepa_clients = [
        FederatedClient(
            i,
            client_data[i],
            "lejepa",
            cfg.num_views,
            device=device,
            dp_config=dp_config,
            img_size=cfg.img_size,
        )
        for i in range(cfg.num_clients)
    ]
    mae_clients = [
        FederatedClient(
            i,
            client_data[i],
            "mae",
            1,
            device=device,
            dp_config=dp_config,
            img_size=cfg.img_size,
        )
        for i in range(cfg.num_clients)
    ]

    lejepa_server = FederatedServer(lejepa_model)
    mae_server = FederatedServer(mae_model)

    lejepa_attacker = GradientInversionAttack(lejepa_model)
    mae_attacker = GradientInversionAttack(mae_model)

    results: Dict[str, Dict[str, List[float]]] = {
        "lejepa": {
            "mse": [],
            "psnr": [],
            "cosine": [],
            "rounds": [],
            "loss": [],
            "loss_rounds": [],
            "probe_acc": [],
            "probe_rounds": [],
            "inv": [],
            "sigreg": [],
        },
        "mae": {
            "mse": [],
            "psnr": [],
            "cosine": [],
            "rounds": [],
            "loss": [],
            "loss_rounds": [],
            "probe_acc": [],
            "probe_rounds": [],
        },
    }

    loss_log_path = "loss_components_log.csv"
    initialize_loss_log(loss_log_path)

    plot_rounds = np.linspace(0, cfg.num_rounds - 1, cfg.plot_rounds, dtype=int)
    last_reconstructions: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for round_idx in range(cfg.num_rounds):
        print(f"\n  Round {round_idx + 1}/{cfg.num_rounds}")

        recon_payload: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

        lejepa_updates = [
            client.local_train(lejepa_server.global_model, epochs=cfg.local_epochs, lr=cfg.lr)
            for client in lejepa_clients
        ]
        lejepa_server.aggregate(lejepa_updates)
        lejepa_avg_loss = float(np.mean([update["loss"] for update in lejepa_updates]))
        results["lejepa"]["loss"].append(lejepa_avg_loss)
        results["lejepa"]["loss_rounds"].append(round_idx)
        results["lejepa"]["inv"].append(float(np.mean([u["inv_loss"] for u in lejepa_updates])))
        results["lejepa"]["sigreg"].append(float(np.mean([u["sigreg_loss"] for u in lejepa_updates])))

        for client_id, update in enumerate(lejepa_updates):
            append_loss_log(
                loss_log_path,
                round_idx,
                "lejepa",
                "client",
                client_id,
                {
                    "total": update["loss"],
                    "inv": update["inv_loss"],
                    "sigreg": update["sigreg_loss"],
                },
            )

        mae_updates = [
            client.local_train(mae_server.global_model, epochs=cfg.local_epochs, lr=cfg.lr)
            for client in mae_clients
        ]
        mae_server.aggregate(mae_updates)
        mae_avg_loss = float(np.mean([update["loss"] for update in mae_updates]))
        results["mae"]["loss"].append(mae_avg_loss)
        results["mae"]["loss_rounds"].append(round_idx)

        for client_id, update in enumerate(mae_updates):
            append_loss_log(
                loss_log_path,
                round_idx,
                "mae",
                "client",
                client_id,
                {"total": update["loss"], "inv": None, "sigreg": None},
            )

        print(
            f"    [Loss] JEPA avg={lejepa_avg_loss:.4f} "
            f"inv={results['lejepa']['inv'][-1]:.4f} sigreg={results['lejepa']['sigreg'][-1]:.4f} | "
            f"MAE avg={mae_avg_loss:.4f}"
        )

        if round_idx % cfg.eval_every == 0 or round_idx == cfg.num_rounds - 1:
            client_idx = 0
            lejepa_grad = lejepa_updates[client_idx]["gradients"]
            mae_grad = mae_updates[client_idx]["gradients"]

            x_test = client_data[client_idx][:8]
            augmenter = ViewAugmenter(num_views=cfg.num_views, img_size=cfg.img_size)
            x_test_views = augmenter(x_test)

            lejepa_recon, _ = lejepa_attacker.attack(
                x_test_views,
                lejepa_grad,
                lr=0.05,
                iterations=200,
                loss_strategy="cosine",
            )
            mae_recon, _ = mae_attacker.attack(
                x_test,
                mae_grad,
                lr=0.05,
                iterations=200,
                loss_strategy="cosine",
            )

            lejepa_metrics = lejepa_attacker.compute_metrics(x_test_views, lejepa_recon)
            mae_metrics = mae_attacker.compute_metrics(x_test, mae_recon)

            results["lejepa"]["mse"].append(lejepa_metrics["mse"])
            results["lejepa"]["psnr"].append(lejepa_metrics["psnr"])
            results["lejepa"]["cosine"].append(lejepa_metrics["cosine_sim"])
            results["lejepa"]["rounds"].append(round_idx)

            results["mae"]["mse"].append(mae_metrics["mse"])
            results["mae"]["psnr"].append(mae_metrics["psnr"])
            results["mae"]["cosine"].append(mae_metrics["cosine_sim"])
            results["mae"]["rounds"].append(round_idx)

            print(
                f"    JEPA: MSE={lejepa_metrics['mse']:.4f}, "
                f"PSNR={lejepa_metrics['psnr']:.2f}dB, Cos={lejepa_metrics['cosine_sim']:.4f}"
            )
            print(
                f"    MAE:  MSE={mae_metrics['mse']:.4f}, "
                f"PSNR={mae_metrics['psnr']:.2f}dB, Cos={mae_metrics['cosine_sim']:.4f}"
            )

            recon_payload = (x_test_views, x_test, lejepa_recon, mae_recon)

            if round_idx == cfg.num_rounds - 1:
                last_reconstructions["lejepa"] = (x_test_views, lejepa_recon)
                last_reconstructions["mae"] = (x_test, mae_recon)

        if round_idx in plot_rounds:
            if recon_payload is None:
                client_idx = 0
                lejepa_grad = lejepa_updates[client_idx]["gradients"]
                mae_grad = mae_updates[client_idx]["gradients"]
                x_test = client_data[client_idx][:8]
                augmenter = ViewAugmenter(num_views=cfg.num_views, img_size=cfg.img_size)
                x_test_views = augmenter(x_test)
                lejepa_recon, _ = lejepa_attacker.attack(
                    x_test_views,
                    lejepa_grad,
                    lr=0.05,
                    iterations=200,
                    loss_strategy="cosine",
                )
                mae_recon, _ = mae_attacker.attack(
                    x_test,
                    mae_grad,
                    lr=0.05,
                    iterations=200,
                    loss_strategy="cosine",
                )
                recon_payload = (x_test_views, x_test, lejepa_recon, mae_recon)

            print(f"\n  [Plotting] Reconstruction grids at round {round_idx + 1}")
            lejepa_orig, mae_orig, lejepa_recon, mae_recon = recon_payload
            plot_reconstructions(
                lejepa_orig,
                lejepa_recon,
                save_path=f"lejepa_reconstructions_round{round_idx + 1}.png",
                title=f"LeJEPA Gradient Inversion (Round {round_idx + 1})",
            )
            plot_reconstructions(
                mae_orig,
                mae_recon,
                save_path=f"mae_reconstructions_round{round_idx + 1}.png",
                title=f"MAE Gradient Inversion (Round {round_idx + 1})",
            )

            print(f"\n  [Probing] Linear probe at round {round_idx + 1}")
            train_data, train_labels = sample_tensor_dataset(
                all_train_data, all_train_labels, max_samples=1500
            )
            test_data, test_labels = sample_tensor_dataset(val_data, val_labels, max_samples=1000)

            lejepa_probe_acc = train_linear_probe_from_tensors(
                lejepa_server.global_model,
                train_data,
                train_labels,
                test_data,
                test_labels,
                model_type="lejepa",
                device=device,
                epochs=5,
                lr=cfg.probe_lr,
            )
            mae_probe_acc = train_linear_probe_from_tensors(
                mae_server.global_model,
                train_data,
                train_labels,
                test_data,
                test_labels,
                model_type="mae",
                device=device,
                epochs=5,
                lr=cfg.probe_lr,
            )
            results["lejepa"]["probe_acc"].append(lejepa_probe_acc)
            results["lejepa"]["probe_rounds"].append(round_idx)
            results["mae"]["probe_acc"].append(mae_probe_acc)
            results["mae"]["probe_rounds"].append(round_idx)
            print(f"    Probe JEPA: {lejepa_probe_acc * 100:.2f}%")
            print(f"    Probe MAE:  {mae_probe_acc * 100:.2f}%")

    if last_reconstructions:
        print("\n[4] Saving reconstructed image grids...")
        lejepa_orig, lejepa_recon = last_reconstructions["lejepa"]
        mae_orig, mae_recon = last_reconstructions["mae"]
        plot_reconstructions(
            lejepa_orig,
            lejepa_recon,
            save_path="lejepa_reconstructions.png",
            title="LeJEPA Gradient Inversion (Cosine)",
        )
        plot_reconstructions(
            mae_orig,
            mae_recon,
            save_path="mae_reconstructions.png",
            title="MAE Gradient Inversion (Cosine)",
        )

    plot_metric_curve(
        results["lejepa"]["loss_rounds"],
        results["lejepa"]["loss"],
        results["mae"]["loss"],
        "LeJEPA",
        "MAE",
        "Local training loss",
        "Federated Training Loss (ImageNette)",
        "training_loss_curve.png",
    )
    plot_metric_curve(
        results["lejepa"]["probe_rounds"],
        results["lejepa"]["probe_acc"],
        results["mae"]["probe_acc"],
        "LeJEPA",
        "MAE",
        "Linear probe accuracy",
        "Linear Probe Accuracy (ImageNette)",
        "linear_probe_curve.png",
    )

    return results


def run(cfg: Config) -> None:
    run_federated_privacy_experiment(cfg)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="LeJEPA vs MAE on ImageNette")
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--lamb", type=float, default=0.02)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--proj-dim", type=int, default=64)
    parser.add_argument("--mae-mask-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--samples-per-client", type=int, default=1000)
    parser.add_argument("--dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--num-rounds", type=int, default=500)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--plot-rounds", type=int, default=5)
    parser.add_argument("--dp-enabled", action="store_true")
    parser.add_argument("--dp-clip-norm", type=float, default=1.0)
    parser.add_argument("--dp-noise-multiplier", type=float, default=0.8)
    args = parser.parse_args()
    return Config(
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        lr=args.lr,
        probe_lr=args.probe_lr,
        weight_decay=args.weight_decay,
        lamb=args.lamb,
        num_views=args.num_views,
        proj_dim=args.proj_dim,
        mae_mask_ratio=args.mae_mask_ratio,
        seed=args.seed,
        num_clients=args.num_clients,
        samples_per_client=args.samples_per_client,
        dirichlet_alpha=args.dirichlet_alpha,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        eval_every=args.eval_every,
        plot_rounds=args.plot_rounds,
        dp_enabled=args.dp_enabled,
        dp_clip_norm=args.dp_clip_norm,
        dp_noise_multiplier=args.dp_noise_multiplier,
    )


if __name__ == "__main__":
    run(parse_args())
