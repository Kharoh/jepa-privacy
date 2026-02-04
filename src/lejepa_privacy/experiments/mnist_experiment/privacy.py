"""Privacy utilities including DP and gradient inversion."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import LeJEPAModel, MAEModel, MAEViT


@dataclass
class DPConfig:
    enabled: bool = False
    clip_norm: float = 1.0
    noise_multiplier: float = 0.8
    seed: int = 42
    apply_to_gradients: bool = True
    apply_to_updates: bool = True


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
    return [
        t + torch.randn(t.shape, device=t.device, dtype=t.dtype, generator=generator) * std
        for t in tensors
    ]


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


class GaussianMIEstimator:
    """
    Estimate I(X; G) assuming Gaussian distributions.
    Uses analytical entropy formulas.
    """

    def __init__(
        self,
        method: str = "gaussian",
        max_dim: int = 256,
        projection_dim: int = 128,
        use_diag: bool = True,
        seed: int = 42,
    ):
        self.method = method
        self.max_dim = max_dim
        self.projection_dim = projection_dim
        self.use_diag = use_diag
        self._proj_cache = {}
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def _reduce_dim(self, samples: torch.Tensor) -> torch.Tensor:
        """Randomly project samples when feature dim is too large."""
        d = samples.shape[1]
        if d <= self.max_dim:
            return samples

        proj_dim = min(self.projection_dim, self.max_dim, d)
        key = (d, proj_dim, samples.device.type)
        if key not in self._proj_cache:
            A = torch.randn(d, proj_dim, generator=self._rng, device=samples.device)
            A = A / np.sqrt(d)
            self._proj_cache[key] = A

        return samples @ self._proj_cache[key]

    def entropy_gaussian(self, samples: torch.Tensor) -> float:
        """H(X) = 0.5 * ln((2*pi*e)^d * |Σ|)"""
        if samples.shape[0] < 2:
            return 0.0

        d = samples.shape[1]

        if self.use_diag or samples.shape[0] <= d:
            var = samples.var(dim=0, unbiased=True) + 1e-6
            logdet = torch.log(var).sum()
            return 0.5 * (d * np.log(2 * np.pi * np.e) + logdet.item())

        centered = samples - samples.mean(dim=0)
        cov = (centered.T @ centered) / (samples.shape[0] - 1)
        cov = cov + torch.eye(d, device=samples.device, dtype=samples.dtype) * 1e-6

        sign, logdet = torch.slogdet(cov)
        if sign.item() <= 0:
            return 0.0

        return 0.5 * (d * np.log(2 * np.pi * np.e) + logdet.item())

    def estimate_mi(self, x: torch.Tensor, g: torch.Tensor) -> float:
        """
        I(X; G) = H(X) + H(G) - H(X, G)
        """
        if g.dim() > 2:
            g = g.reshape(g.shape[0], -1)
        elif g.dim() == 1:
            g = g.unsqueeze(0)

        if g.shape[0] != x.shape[0]:
            if g.shape[0] == 1:
                g = g.repeat(x.shape[0], 1)
            elif g.shape[0] < x.shape[0]:
                repeats = int(np.ceil(x.shape[0] / g.shape[0]))
                g = g.repeat(repeats, 1)[: x.shape[0]]
            else:
                g = g[: x.shape[0]]

        x_reduced = self._reduce_dim(x)
        g_reduced = self._reduce_dim(g)

        joint = torch.cat([x_reduced, g_reduced], dim=1)

        H_joint = self.entropy_gaussian(joint)
        H_x = self.entropy_gaussian(x_reduced)
        H_g = self.entropy_gaussian(g_reduced)

        mi = H_x + H_g - H_joint
        return max(0, mi)


class MINEEstimator(nn.Module):
    """
    MINE: Mutual Information Neural Estimation.
    Uses Donsker-Varadhan representation.
    """

    def __init__(self, x_dim: int, g_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.T = nn.Sequential(
            nn.Linear(x_dim + g_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Negative DV bound (we minimize this)."""
        joint = torch.cat([x, g], dim=1)
        T_joint = self.T(joint)

        g_shuffled = g[torch.randperm(g.shape[0])]
        marginal = torch.cat([x, g_shuffled], dim=1)
        T_marginal = torch.exp(self.T(marginal))

        return -(T_joint.mean() - torch.log(T_marginal.mean()))

    def estimate_mi(self, x: torch.Tensor, g: torch.Tensor, iters: int = 200) -> float:
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)

        for _ in range(iters):
            opt.zero_grad()
            loss = self.forward(x, g)
            loss.backward()
            opt.step()

        with torch.no_grad():
            mi = -self.forward(x, g).item()
        return mi


class GradientInversionAttack:
    """
    Reconstruct data from gradients to measure privacy leakage.
    Improved with Cosine Similarity loss and Total Variation regularization.
    """

    def __init__(self, model: nn.Module, input_dim: int, num_views: int = 2):
        self.model = model
        self.input_dim = input_dim
        self.num_views = num_views

    def attack(
        self,
        original_data: torch.Tensor,
        true_grad: torch.Tensor,
        iterations: int = 500,
        lr: float = 0.1,
        return_history: bool = False,
        record_steps: List[int] | None = None,
        loss_strategy: str = "cosine",
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Reconstruct data from gradients using optimization.
        """
        device = next(self.model.parameters()).device
        original_data = original_data.to(device)
        true_grad = true_grad.to(device)

        B = original_data.shape[0]
        dummy = (torch.randn(B, self.num_views, self.input_dim, device=device) * 0.1).requires_grad_(True)

        opt = torch.optim.Adam([dummy], lr=lr)

        tv_weight = 1e-4

        true_grad = true_grad.detach()

        history: Dict[int, torch.Tensor] = {}
        record_steps = set(record_steps or [])
        if 0 in record_steps:
            history[0] = dummy.detach().cpu().clone()

        loss_strategy = loss_strategy.lower().strip()

        for step in range(1, iterations + 1):
            opt.zero_grad()
            self.model.zero_grad(set_to_none=True)

            if isinstance(self.model, LeJEPAModel):
                loss = self.model.compute_loss(dummy)["total"]
            elif isinstance(self.model, (MAEModel, MAEViT)):
                loss = self.model.reconstruction_loss(dummy)
            else:
                raise ValueError("Unknown model type for gradient inversion")

            params = [p for p in self.model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(
                loss,
                params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            grads_flat = [g.reshape(-1) for g in grads if g is not None]
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

            img_size = int(np.sqrt(self.input_dim))
            if img_size * img_size == self.input_dim:
                dummy_img = dummy.view(-1, 1, img_size, img_size)
                h_diff = dummy_img[:, :, 1:, :] - dummy_img[:, :, :-1, :]
                w_diff = dummy_img[:, :, :, 1:] - dummy_img[:, :, :, :-1]
                tv_loss = torch.sum(torch.abs(h_diff)) + torch.sum(torch.abs(w_diff))
            else:
                tv_loss = torch.tensor(0.0, device=device)

            total_loss = grad_loss + (tv_weight * tv_loss)

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            if step in record_steps:
                history[step] = dummy.detach().cpu().clone()

        if return_history:
            return dummy.detach().cpu(), history
        return dummy.detach().cpu(), {}

    def compute_metrics(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        denormalize_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        data_range: float = 1.0,
        clip_range: Tuple[float, float] = (0.0, 1.0),
    ) -> Dict[str, float]:
        """Compute privacy leakage metrics."""
        if original.dim() == 3:
            original = original[:, 0, :]

        if reconstructed.dim() == 3:
            reconstructed = reconstructed[:, 0, :]

        if original.device != reconstructed.device:
            reconstructed = reconstructed.cpu()
        original = original.cpu()

        if denormalize_fn is not None:
            original = denormalize_fn(original)
            reconstructed = denormalize_fn(reconstructed)

        if clip_range is not None:
            low, high = clip_range
            original = original.clamp(low, high)
            reconstructed = reconstructed.clamp(low, high)

        metrics: Dict[str, float] = {}

        metrics["mse"] = F.mse_loss(reconstructed, original).item()

        mse = metrics["mse"]
        metrics["psnr"] = 10 * np.log10((data_range**2) / (mse + 1e-10))

        orig_flat = original.flatten()
        recon_flat = reconstructed.flatten()
        metrics["cosine_sim"] = F.cosine_similarity(
            orig_flat.unsqueeze(0), recon_flat.unsqueeze(0)
        ).item()

        metrics["rel_l2"] = torch.norm(original - reconstructed).item() / (
            torch.norm(original).item() + 1e-10
        )

        orig_centered = original - original.mean(dim=0)
        recon_centered = reconstructed - reconstructed.mean(dim=0)
        metrics["correlation"] = (orig_centered * recon_centered).sum().item() / (
            torch.norm(orig_centered) * torch.norm(recon_centered) + 1e-10
        ).item()

        return metrics


class UpdateInversionAttack:
    """
    Reconstruct data from model updates (FedAvg-style) by matching a one-step
    SGD update computed on dummy data.
    """

    def __init__(
        self,
        model: nn.Module,
        input_dim: int,
        num_views: int = 2,
        view_augmenter: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.model = model
        self.input_dim = input_dim
        self.num_views = num_views
        self.view_augmenter = view_augmenter

    def _prepare_views(self, dummy: torch.Tensor, model: nn.Module) -> torch.Tensor:
        if isinstance(model, LeJEPAModel):
            if dummy.dim() == 2:
                if self.view_augmenter is None:
                    raise ValueError("view_augmenter required for LeJEPA update inversion")
                return self.view_augmenter(dummy.clone())
            return dummy
        if dummy.dim() == 3:
            return dummy[:, 0, :]
        if self.view_augmenter is not None and dummy.dim() == 2:
            aug = self.view_augmenter(dummy.clone())
            if aug.dim() == 3:
                return aug[:, 0, :]
            return aug
        return dummy

    def _compute_update_vector(self, dummy: torch.Tensor, lr: float) -> torch.Tensor:
        model_copy = copy.deepcopy(self.model)
        model_copy.zero_grad(set_to_none=True)
        model_copy.train()

        views = self._prepare_views(dummy, model_copy)
        if isinstance(model_copy, LeJEPAModel):
            loss = model_copy.compute_loss(views)["total"]
        elif isinstance(model_copy, (MAEModel, MAEViT)):
            loss = model_copy.reconstruction_loss(views)
        else:
            raise ValueError("Unknown model type for update inversion")

        params = [p for p in model_copy.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

        updates = [(-lr * grad).flatten() for grad in grads if grad is not None]
        if not updates:
            raise RuntimeError("No gradients found for update inversion")
        return torch.cat(updates)

    def attack(
        self,
        original_data: torch.Tensor,
        true_update: torch.Tensor,
        iterations: int = 500,
        lr: float = 0.1,
        update_lr: float = 1e-3,
        return_history: bool = False,
        record_steps: List[int] | None = None,
        loss_strategy: str = "cosine",
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        device = next(self.model.parameters()).device
        original_data = original_data.to(device)
        true_update = true_update.to(device)

        b = original_data.shape[0]
        dummy = (torch.randn(b, self.input_dim, device=device) * 0.1).requires_grad_(True)
        opt = torch.optim.Adam([dummy], lr=lr)

        history: Dict[int, torch.Tensor] = {}
        record_steps = set(record_steps or [])
        if 0 in record_steps:
            history[0] = dummy.detach().cpu().clone()

        loss_strategy = loss_strategy.lower().strip()

        for step in range(1, iterations + 1):
            opt.zero_grad()
            update_vec = self._compute_update_vector(dummy, lr=update_lr)

            if loss_strategy == "cosine":
                update_loss = 1.0 - F.cosine_similarity(
                    update_vec.unsqueeze(0), true_update.unsqueeze(0)
                )
            elif loss_strategy == "mse":
                update_loss = F.mse_loss(update_vec, true_update)
            else:
                raise ValueError(f"Unknown loss_strategy: {loss_strategy}")

            total_loss = update_loss
            total_loss.backward()
            opt.step()

            if step in record_steps:
                history[step] = dummy.detach().cpu().clone()

        if return_history:
            return dummy.detach().cpu(), history
        return dummy.detach().cpu(), {}


def clone_model(model: nn.Module) -> nn.Module:
    cloned = copy.deepcopy(model)
    return cloned
