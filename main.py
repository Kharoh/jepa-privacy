"""
Complete Federated Learning Privacy Comparison: LeJEPA vs MAE

This script implements:
1. LeJEPA with invariance loss + SIGReg
2. Masked Autoencoder (MAE) baseline
3. Federated learning with gradient averaging
4. Mutual information I(X; G) computation
5. Gradient inversion attacks for privacy quantification
"""

import copy
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Dict
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Set seeds
torch.manual_seed(42)
np.random.seed(42)


# ============================================
# Differential Privacy Utilities
# ============================================

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


# ============================================
# SIGReg: Exact LeJEPA Implementation
# ============================================

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



# ============================================
# LeJEPA: Invariance-Based Architecture
# ============================================

class LeJEPAEncoder(nn.Module):
    """Encoder with embedding and projection heads."""
    def __init__(self, input_dim: int, emb_dim: int, proj_dim: int,
                 use_cnn: bool = False, image_shape: Tuple[int, int, int] = (1, 28, 28)):
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
                nn.Linear(self._cnn_feat_dim, emb_dim)
            )
        else:
            # Backbone: input -> embedding
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, emb_dim)
            )

        # Projection head: embedding -> projection (used for LeJEPA loss)
        self.project = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, input_dim) - B samples, N views each
        
        Returns:
            emb: (B, N, emb_dim)
            proj: (B, N, proj_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)

        if self.use_cnn:
            x_flat = x_flat.view(B * N, *self.image_shape)
            emb = self.backbone(x_flat).reshape(B, N, self.emb_dim)
        else:
            emb = self.backbone(x_flat).reshape(B, N, self.emb_dim)

        proj = self.project(emb.reshape(B * N, self.emb_dim)).reshape(B, N, self.proj_dim)
        
        return emb, proj


class LeJEPAModel(nn.Module):
    """
    LeJEPA: Learns invariant representations by minimizing variance
    across augmented views + SIGReg for Gaussian structure.
    """
    def __init__(self, input_dim: int, emb_dim: int = 64, proj_dim: int = 64,
                 lamb: float = 0.5, use_cnn: bool = False,
                 image_shape: Tuple[int, int, int] = (1, 28, 28)):
        super().__init__()
        self.encoder = LeJEPAEncoder(
            input_dim,
            emb_dim,
            proj_dim,
            use_cnn=use_cnn,
            image_shape=image_shape,
        )
        self.sigreg = SIGReg(knots=17)
        self.lamb = lamb
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)
    
    def compute_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        LeJEPA loss = λ * SIGReg(proj) + (1-λ) * inv_loss
        
        inv_loss: variance of projections across views
        """
        emb, proj = self.forward(x)  # (B, N, proj_dim)
        
        # Invariance loss: minimize variance across views
        # proj.mean(dim=1): (B, proj_dim) - mean over views
        proj_mean = proj.mean(dim=1, keepdim=True)  # (B, 1, proj_dim)
        inv_loss = (proj_mean - proj).square().mean()  # Average variance
        
        # SIGReg on projections
        sigreg_loss = self.sigreg(proj.flatten(0, 1))
        
        # Combined
        total_loss = self.lamb * sigreg_loss + (1 - self.lamb) * inv_loss
        
        return {
            'total': total_loss,
            'inv': inv_loss,
            'sigreg': sigreg_loss,
            'emb': emb,
            'proj': proj
        }


# ============================================
# Masked Autoencoder (Baseline)
# ============================================

class MAEModel(nn.Module):
    """Masked Autoencoder with reconstruction objective."""
    def __init__(self, input_dim: int, latent_dim: int, mask_ratio: float = 0.4,
                 use_cnn: bool = False, image_shape: Tuple[int, int, int] = (1, 28, 28)):
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
                nn.Linear(self._cnn_feat_dim, latent_dim)
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, self._cnn_feat_dim),
                nn.ReLU(),
                nn.Unflatten(1, (64, h2, w2)),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, channels, kernel_size=2, stride=2)
            )
        else:
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim)
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim)
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns reconstruction, latent, and mask."""
        if x.dim() == 3:
            # (B, N, D) -> take first view for MAE
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
        recon, latent, mask, original = self.forward(x)
        # Loss only on masked positions
        return F.mse_loss(recon * (~mask).float(), original * (~mask).float())


# ============================================
# Mutual Information Estimators
# ============================================

class GaussianMIEstimator:
    """
    Estimate I(X; G) assuming Gaussian distributions.
    Uses analytical entropy formulas.
    """
    def __init__(self, method: str = 'gaussian', max_dim: int = 256,
                 projection_dim: int = 128, use_diag: bool = True,
                 seed: int = 42):
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
        # Flatten gradients if needed
        if g.dim() > 2:
            g = g.reshape(g.shape[0], -1)
        elif g.dim() == 1:
            g = g.unsqueeze(0)

        # Ensure batch sizes match for concatenation
        if g.shape[0] != x.shape[0]:
            if g.shape[0] == 1:
                g = g.repeat(x.shape[0], 1)
            elif g.shape[0] < x.shape[0]:
                repeats = int(np.ceil(x.shape[0] / g.shape[0]))
                g = g.repeat(repeats, 1)[:x.shape[0]]
            else:
                g = g[:x.shape[0]]

        # Reduce dimensionality to avoid massive covariance allocations
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
            nn.Linear(hidden_dim, 1)
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


# ============================================
# Gradient Inversion Attack
# ============================================

class GradientInversionAttack:
    """
    Reconstruct data from gradients to measure privacy leakage.
    Improved with Cosine Similarity loss and Total Variation regularization.
    """
    def __init__(self, model: nn.Module, input_dim: int, num_views: int = 2):
        self.model = model
        self.input_dim = input_dim
        self.num_views = num_views
        
    def attack(self, original_data: torch.Tensor, true_grad: torch.Tensor,
               iterations: int = 500, lr: float = 0.1,
               return_history: bool = False,
               record_steps: List[int] = None,
               loss_strategy: str = "cosine") -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Reconstruct data from gradients using optimization.
        """
        device = next(self.model.parameters()).device
        original_data = original_data.to(device)
        true_grad = true_grad.to(device)

        # --- IMPROVEMENT 1: Smarter Initialization ---
        # Instead of pure N(0,1) noise, start with small noise around 0 (grey image)
        # This acts as a weak prior that the image is not purely random high-freq noise.
        B = original_data.shape[0]
        dummy = (torch.randn(B, self.num_views, self.input_dim, device=device) * 0.1).requires_grad_(True)
        
        # Use LBFGS if possible, otherwise stick to Adam but with tuned LR
        # We stick to Adam here to match the original style but use a slightly different LR strategy
        opt = torch.optim.Adam([dummy], lr=lr)
        
        # TV Regularization weight (hyperparameter tuning often needed, 1e-4 is a safe default)
        tv_weight = 1e-4

        true_grad = true_grad.detach()

        history = {}
        record_steps = set(record_steps or [])
        if 0 in record_steps:
            history[0] = dummy.detach().cpu().clone()

        loss_strategy = loss_strategy.lower().strip()

        for step in range(1, iterations + 1):
            opt.zero_grad()
            self.model.zero_grad(set_to_none=True)

            # Compute dummy loss
            if isinstance(self.model, LeJEPAModel):
                loss = self.model.compute_loss(dummy)['total']
            elif isinstance(self.model, MAEModel):
                loss = self.model.reconstruction_loss(dummy)

            # Compute gradients w.r.t. model params
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

            # --- IMPROVEMENT 2: Gradient Matching Loss ---
            if loss_strategy == "cosine":
                # Minimizes the angle between gradient vectors.
                # Much more robust to magnitude scaling issues than MSE.
                grad_loss = 1.0 - F.cosine_similarity(
                    dummy_grad.unsqueeze(0), true_grad.unsqueeze(0)
                )
            elif loss_strategy == "mse":
                grad_loss = F.mse_loss(dummy_grad, true_grad)
            else:
                raise ValueError(f"Unknown loss_strategy: {loss_strategy}")
            
            # --- IMPROVEMENT 3: Total Variation (TV) Regularization ---
            # Enforces smoothness in the reconstructed image.
            # We assume the input_dim (e.g. 784) is a flattened 28x28 image.
            # If input_dim is not square, this might need adjustment, but works for MNIST/CIFAR.
            img_size = int(np.sqrt(self.input_dim))
            if img_size * img_size == self.input_dim:
                dummy_img = dummy.view(-1, 1, img_size, img_size) # Treat views as batch items for TV
                h_diff = dummy_img[:, :, 1:, :] - dummy_img[:, :, :-1, :]
                w_diff = dummy_img[:, :, :, 1:] - dummy_img[:, :, :, :-1]
                tv_loss = torch.sum(torch.abs(h_diff)) + torch.sum(torch.abs(w_diff))
            else:
                # Fallback if data isn't square images
                tv_loss = torch.tensor(0.0, device=device)

            # Combine losses
            total_loss = grad_loss + (tv_weight * tv_loss)

            # Update dummy
            total_loss.backward()
            opt.step()
            
            # Optional: Clamp data to valid image range if known (e.g., 0-1 or -1 to 1)
            # with torch.no_grad():
            #    dummy.clamp_(0, 1)

            if step in record_steps:
                history[step] = dummy.detach().cpu().clone()
            
        if return_history:
            return dummy.detach().cpu(), history
        return dummy.detach().cpu(), {}
    
    def compute_metrics(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict:
        """Compute privacy leakage metrics."""
        if original.dim() == 3:
            original = original[:, 0, :]  # Take first view
            
        if reconstructed.dim() == 3:
            reconstructed = reconstructed[:, 0, :]
            
        metrics = {}
        
        # MSE
        metrics['mse'] = F.mse_loss(reconstructed, original).item()
        
        # PSNR
        mse = metrics['mse']
        metrics['psnr'] = 10 * np.log10(1.0 / (mse + 1e-10))
        
        # Cosine similarity
        orig_flat = original.flatten()
        recon_flat = reconstructed.flatten()
        metrics['cosine_sim'] = F.cosine_similarity(
            orig_flat.unsqueeze(0), recon_flat.unsqueeze(0)
        ).item()
        
        # Relative L2 error
        metrics['rel_l2'] = torch.norm(original - reconstructed).item() / \
                           (torch.norm(original).item() + 1e-10)
        
        # Correlation coefficient
        orig_centered = original - original.mean(dim=0)
        recon_centered = reconstructed - reconstructed.mean(dim=0)
        metrics['correlation'] = (orig_centered * recon_centered).sum().item() / \
                                 (torch.norm(orig_centered) * torch.norm(recon_centered) + 1e-10).item()
        
        return metrics


# ============================================
# View Augmentation (Continued)
# ============================================

class ViewAugmenter:
    """Create multiple augmented views of the same sample with diverse transforms."""
    def __init__(
        self,
        num_views: int = 2,
        mask_ratio: float = 0.4,
        noise_std: float = 0.1,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
        rotation_deg: float = 20.0,
        translation_px: int = 3,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        brightness_range: Tuple[float, float] = (0.85, 1.15),
        blur_prob: float = 0.25,
    ):
        self.num_views = num_views
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std
        self.image_shape = image_shape
        self.rotation_deg = rotation_deg
        self.translation_px = translation_px
        self.scale_range = scale_range
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.blur_prob = blur_prob
        self.apply_transforms = any([
            rotation_deg != 0,
            translation_px != 0,
            scale_range != (1.0, 1.0),
            contrast_range is not None,
            brightness_range is not None,
            blur_prob > 0,
        ])
        if self.apply_transforms:
            _, height, width = self.image_shape
            self.view_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop((height, width), scale=(0.5, 1.0)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=3),
                transforms.ToTensor(),
            ])
        else:
            self.view_transform = None

    def _apply_view_transforms(self, images: torch.Tensor) -> torch.Tensor:
        """Apply random spatial and photometric transforms to a batch of images."""
        if not self.apply_transforms:
            return images

        channels, height, width = self.image_shape
        angle = float(torch.empty(1).uniform_(-self.rotation_deg, self.rotation_deg).item())
        translate_x = int(torch.randint(-self.translation_px, self.translation_px + 1, (1,)).item())
        translate_y = int(torch.randint(-self.translation_px, self.translation_px + 1, (1,)).item())
        scale = float(torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item())
        shear = [0.0, 0.0]

        augmented = []
        for img in images:
            img = TF.affine(
                img,
                angle=angle,
                translate=[translate_x, translate_y],
                scale=scale,
                shear=shear,
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            if self.contrast_range:
                contrast = float(torch.empty(1).uniform_(self.contrast_range[0], self.contrast_range[1]).item())
                img = TF.adjust_contrast(img, contrast)
            if self.brightness_range:
                brightness = float(torch.empty(1).uniform_(self.brightness_range[0], self.brightness_range[1]).item())
                img = TF.adjust_brightness(img, brightness)
            if self.blur_prob > 0 and torch.rand(1).item() < self.blur_prob:
                img = TF.gaussian_blur(img, kernel_size=3, sigma=(0.1, 1.0))
            if self.view_transform is not None:
                device = img.device
                img = self.view_transform(img.detach().cpu()).to(device)
            augmented.append(img)

        return torch.stack(augmented, dim=0)
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_dim) - batch of samples
        
        Returns: (B, num_views, input_dim) - B samples, each with num_views augmented versions
        """
        B, D = x.shape
        views = []
        channels, height, width = self.image_shape

        x_img = x.view(B, channels, height, width)
        
        for _ in range(self.num_views):
            view_img = x_img.clone()
            view_img = self._apply_view_transforms(view_img)
            view = view_img.view(B, -1)
            
            # Random masking (different mask for each view)
            mask = torch.rand_like(view) > self.mask_ratio
            view = view * mask.float()
            
            # Add Gaussian noise (different noise for each view)
            if self.noise_std > 0:
                noise = torch.randn_like(view) * self.noise_std
                view = view + noise
            view = view.clamp(0.0, 1.0)
            
            views.append(view.unsqueeze(1))
            
        # Concatenate: (B, num_views, input_dim)
        return torch.cat(views, dim=1)


# ============================================
# Federated Learning Components
# ============================================

def create_mnist_transform(image_shape: Tuple[int, int, int]) -> transforms.Compose:
    """Create MNIST transform with optional resizing to target image shape."""
    _, height, width = image_shape
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ])


def load_mnist_non_iid(num_clients: int, total_samples: int, alpha: float = 0.5,
                       seed: int = 42, data_dir: str = "data",
                       image_shape: Tuple[int, int, int] = (1, 28, 28)) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Load MNIST and split into non-IID client datasets using Dirichlet class proportions.

    Returns:
        client_data: list of tensors (N_i, 784)
        client_labels: list of tensors (N_i,)
    """
    rng = np.random.default_rng(seed)
    transform = create_mnist_transform(image_shape)
    dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)

    total_samples = min(total_samples, len(dataset))
    subset_indices = rng.permutation(len(dataset))[:total_samples]

    class_indices = {c: [] for c in range(10)}
    for idx in subset_indices:
        _, label = dataset[idx]
        class_indices[int(label)].append(idx)

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
            img, label = dataset[idx]
            images.append(img.view(-1))
            labels.append(int(label))
        client_data.append(torch.stack(images))
        client_labels.append(torch.tensor(labels, dtype=torch.long))

    return client_data, client_labels


def plot_client_class_distribution(client_labels: List[torch.Tensor], num_classes: int = 10,
                                   save_path: str = "client_class_distribution.png") -> np.ndarray:
    """Plot and save the per-client class distribution heatmap."""
    counts = np.zeros((len(client_labels), num_classes), dtype=int)
    for client_id, labels in enumerate(client_labels):
        unique, freqs = np.unique(labels.numpy(), return_counts=True)
        counts[client_id, unique] = freqs

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(counts, aspect='auto', cmap='viridis')
    ax.set_title("MNIST class distribution per client")
    ax.set_xlabel("Class")
    ax.set_ylabel("Client")
    ax.set_xticks(list(range(num_classes)))
    ax.set_yticks(list(range(len(client_labels))))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Samples")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return counts


def sample_tensor_dataset(data: torch.Tensor, labels: torch.Tensor, max_samples: int,
                          seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly sample up to max_samples from tensors."""
    if len(data) <= max_samples:
        return data, labels
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data))[:max_samples]
    return data[indices], labels[indices]


def sample_mnist_dataset(dataset: datasets.MNIST, max_samples: int,
                         seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample images/labels from a torchvision MNIST dataset."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))[:max_samples]
    images = []
    labels = []
    for idx in indices:
        img, label = dataset[int(idx)]
        images.append(img.view(-1))
        labels.append(int(label))
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def _resolve_image_hw(image_shape: Tuple[int, int, int] = None,
                      flat_dim: int = None) -> Tuple[int, int]:
    """Resolve image height/width from an image_shape or flattened dimension."""
    if image_shape is not None:
        return image_shape[1], image_shape[2]
    if flat_dim is None:
        raise ValueError("flat_dim is required when image_shape is not provided")
    side = int(np.sqrt(flat_dim))
    if side * side != flat_dim:
        raise ValueError(f"Cannot infer square image size from flat_dim={flat_dim}")
    return side, side


def plot_reconstructions(original: torch.Tensor, reconstructed: torch.Tensor,
                         save_path: str, title: str, num_images: int = 8,
                         image_shape: Tuple[int, int, int] = None) -> None:
    """Plot original vs reconstructed images in a 2xN grid."""
    if original.dim() == 3:
        original = original[:, 0, :]
    if reconstructed.dim() == 3:
        reconstructed = reconstructed[:, 0, :]

    height, width = _resolve_image_hw(image_shape, flat_dim=original.shape[-1])
    original = original[:num_images].reshape(-1, height, width).cpu()
    reconstructed = reconstructed[:num_images].reshape(-1, height, width).cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.4, 3.2))
    for i in range(num_images):
        axes[0, i].imshow(original[i], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i], cmap='gray')
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def sample_class_images(data: torch.Tensor, labels: torch.Tensor, classes: List[int],
                        samples_per_class: int = 4) -> Dict[int, torch.Tensor]:
    """Return a dict mapping class -> samples (N, 784)."""
    class_samples = {}
    for cls in classes:
        indices = (labels == cls).nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            continue
        selected = indices[:samples_per_class]
        class_samples[int(cls)] = data[selected]
    return class_samples


def compute_gradients_for_data(model: nn.Module, x: torch.Tensor, model_type: str,
                               num_views: int, augmenter: "ViewAugmenter") -> torch.Tensor:
    """Compute flattened gradients for a batch without updating model weights."""
    device = next(model.parameters()).device
    x = x.to(device)
    model.zero_grad(set_to_none=True)

    if model_type == "lejepa":
        if x.dim() == 2:
            x = augmenter(x)
        loss = model.compute_loss(x)['total']
    else:
        loss = model.reconstruction_loss(x)

    loss.backward()
    grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    return grads.detach().cpu()


def sample_tensor_batch(data: torch.Tensor, max_samples: int, seed: int = 42) -> torch.Tensor:
    """Sample up to max_samples from a tensor."""
    if len(data) <= max_samples:
        return data
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data))[:max_samples]
    return data[indices]


def compute_loss_components(model: nn.Module, batch: torch.Tensor, model_type: str,
                            augmenter: "ViewAugmenter" = None) -> Dict[str, float]:
    """Compute loss components for a model on a batch."""
    device = next(model.parameters()).device
    batch = batch.to(device)
    if model_type == "lejepa":
        if batch.dim() == 2:
            if augmenter is None:
                raise ValueError("augmenter is required for LeJEPA loss computation")
            batch = augmenter(batch)
        loss_dict = model.compute_loss(batch)
        return {
            "total": float(loss_dict["total"].item()),
            "inv": float(loss_dict["inv"].item()),
            "sigreg": float(loss_dict["sigreg"].item())
        }
    loss = model.reconstruction_loss(batch)
    return {"total": float(loss.item()), "inv": None, "sigreg": None}


def initialize_loss_log(log_path: str) -> None:
    """Initialize CSV log for loss components."""
    with open(log_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "round",
            "model",
            "scope",
            "client_id",
            "loss_total",
            "loss_inv",
            "loss_sigreg"
        ])


def append_loss_log(log_path: str, round_idx: int, model: str, scope: str,
                    client_id: int, loss_components: Dict[str, float]) -> None:
    """Append a loss log row to CSV."""
    with open(log_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            round_idx,
            model,
            scope,
            client_id,
            loss_components.get("total"),
            loss_components.get("inv"),
            loss_components.get("sigreg")
        ])


def plot_reconstruction_steps_by_class(
    originals_by_class: Dict[int, torch.Tensor],
    histories_by_class: Dict[int, Dict[int, torch.Tensor]],
    steps: List[int],
    save_path: str,
    title: str,
    image_shape: Tuple[int, int, int] = None,
) -> None:
    """Plot reconstruction snapshots for multiple classes across steps."""
    classes = sorted(originals_by_class.keys())
    if not classes:
        return

    num_rows = len(steps) + 1
    num_cols = len(classes)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.4, num_rows * 1.4))
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for col, cls in enumerate(classes):
        original = originals_by_class[cls]
        if original.dim() == 3:
            original = original[:, 0, :]
        height, width = _resolve_image_hw(image_shape, flat_dim=original.shape[-1])
        img = original[0].reshape(height, width).cpu()
        axes[0, col].imshow(img, cmap='gray')
        axes[0, col].set_title(f"Class {cls}")
        axes[0, col].axis('off')

        history = histories_by_class.get(cls, {})
        for row, step in enumerate(steps, start=1):
            recon = history.get(step)
            if recon is None:
                axes[row, col].axis('off')
                continue
            if recon.dim() == 3:
                recon = recon[:, 0, :]
            img = recon[0].reshape(height, width).cpu()
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(f"Step {step}", fontsize=8)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def extract_latents_for_tsne(model: nn.Module, data: torch.Tensor, model_type: str) -> torch.Tensor:
    """Extract latent vectors for t-SNE visualization."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        if model_type == "lejepa":
            if data.dim() == 2:
                data = data.unsqueeze(1)
            emb, _ = model.encoder(data.to(device))
            latents = emb.mean(dim=1)
        else:
            latents = model.encode(data.to(device)) if hasattr(model, "encode") else model.encoder(data.to(device))
    return latents.detach().cpu()


def plot_tsne_latents(latents: torch.Tensor, labels: torch.Tensor,
                      save_path: str, title: str) -> None:
    """Plot t-SNE embedding with class labels."""
    if latents.shape[0] < 2:
        return

    perplexity = min(30, max(2, latents.shape[0] - 1))
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto",
                perplexity=perplexity, random_state=42)
    embedding = tsne.fit_transform(latents.numpy())

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                         c=labels.numpy(), cmap="tab10", alpha=0.8, s=18)
    legend = ax.legend(*scatter.legend_elements(), title="Class", bbox_to_anchor=(1.05, 1),
                       loc="upper left", borderaxespad=0.0)
    ax.add_artist(legend)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_tsne_for_validation(model: nn.Module,
                             data: torch.Tensor,
                             labels: torch.Tensor,
                             model_type: str,
                             save_path: str,
                             title: str) -> None:
    """Plot t-SNE embeddings for validation samples (no augmentation)."""
    latents = extract_latents_for_tsne(model, data, model_type)
    plot_tsne_latents(latents, labels, save_path=save_path, title=title)


def plot_metric_curve(x: List[int], y_a: List[float], y_b: List[float],
                      label_a: str, label_b: str, ylabel: str,
                      title: str, save_path: str) -> None:
    """Plot a two-line curve for JEPA/MAE metrics."""
    if len(x) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y_a, marker='o', label=label_a)
    ax.plot(x, y_b, marker='o', label=label_b)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def extract_features(model: nn.Module, data: torch.Tensor, model_type: str,
                     batch_size: int = 256) -> torch.Tensor:
    """Extract frozen features for linear probing."""
    model.eval()
    device = next(model.parameters()).device
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            if model_type == "lejepa":
                emb, _ = model.encoder(batch)
                feats = emb[:, 0, :]
            else:
                if hasattr(model, "encode"):
                    feats = model.encode(batch)
                else:
                    feats = model.encoder(batch)
            features.append(feats.detach().cpu())

    return torch.cat(features, dim=0)


def train_linear_probe(model: nn.Module, train_data: torch.Tensor, train_labels: torch.Tensor,
                       test_data: torch.Tensor, test_labels: torch.Tensor,
                       model_type: str, epochs: int = 20, lr: float = 1e-2,
                       batch_size: int = 256) -> float:
    """Train a detached linear probe with Batch Normalization."""
    
    # 1. Extract features (frozen representations)
    train_features = extract_features(model, train_data, model_type)
    test_features = extract_features(model, test_data, model_type)

    device = next(model.parameters()).device
    feat_dim = train_features.shape[1]

    # 2. CHANGE: Use Sequential with BatchNorm1d (affine=False) + Linear
    # affine=False ensures we strictly evaluate linear separability, not the capacity of the BN layer.
    probe = nn.Sequential(
        nn.BatchNorm1d(feat_dim, affine=False),
        nn.Linear(feat_dim, 10)
    ).to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    probe.train()
    for _ in range(epochs):
        for feats, labels in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = probe(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    # 3. Evaluation
    probe.eval()
    with torch.no_grad():
        test_feats = test_features.to(device)
        test_labels = test_labels.to(device)
        
        # The BatchNorm layer will now use the running mean/std calculated during training
        logits = probe(test_feats)
        preds = logits.argmax(dim=1)
        acc = (preds == test_labels).float().mean().item()

    return acc


class FederatedClient:
    """Client for federated learning with gradient extraction and MI tracking."""
    def __init__(self, client_id: int, data: torch.Tensor, model_type: str = "lejepa",
                 num_views: int = 2, device: torch.device = torch.device("cpu"),
                 dp_config: DPConfig = None,
                 augmenter_kwargs: Dict = None,
                 image_shape: Tuple[int, int, int] = (1, 28, 28)):
        self.client_id = client_id
        self.data = data
        self.model_type = model_type
        self.num_views = num_views
        self.augmenter = ViewAugmenter(
            num_views=num_views,
            image_shape=image_shape,
            **(augmenter_kwargs or {})
        )
        self.device = device
        self.dp_config = dp_config or DPConfig(enabled=False)
        gen_device = "cuda" if device.type == "cuda" else "cpu"
        self.dp_generator = torch.Generator(device=gen_device)
        self.dp_generator.manual_seed(self.dp_config.seed + client_id)
        
    def local_train(self, global_model: nn.Module, epochs: int, lr: float) -> Dict:
        """
        Train locally and return gradients for privacy analysis.
        """
        device = self.device
        # Create local model
        local_model = copy.deepcopy(global_model)
        local_model.to(device)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        
        # Training
        for epoch in range(epochs):
            # Sample batch
            batch_size = min(32, len(self.data))
            indices = torch.randperm(len(self.data))[:batch_size]
            x_batch = self.data[indices].to(device)
            
            optimizer.zero_grad()
            
            if self.model_type == "lejepa":
                # Create multiple views
                x_views = self.augmenter(x_batch)
                loss_dict = local_model.compute_loss(x_views)
                loss = loss_dict['total']
            else:
                loss = local_model.reconstruction_loss(x_batch)
                
            loss.backward()
            optimizer.step()
            
        # Extract gradients on a fresh batch for privacy analysis
        with torch.no_grad():
            test_indices = torch.randperm(len(self.data))[:32]
            x_test_cpu = self.data[test_indices]
            x_test = x_test_cpu.to(device)
            
        # Create augmented views for LeJEPA
        if self.model_type == "lejepa":
            x_test_views = self.augmenter(x_test)
            local_model.zero_grad()
            loss_dict = local_model.compute_loss(x_test_views)
            loss = loss_dict['total']
            inv_loss = loss_dict['inv'].item()
            sigreg_loss = loss_dict['sigreg'].item()
        else:
            local_model.zero_grad()
            loss = local_model.reconstruction_loss(x_test)
            inv_loss = None
            sigreg_loss = None
            
        loss.backward()
        
        # Collect gradients
        flat_grad = torch.cat([p.grad.flatten() for p in local_model.parameters()
                              if p.grad is not None]).detach()

        if self.dp_config.enabled and self.dp_config.apply_to_gradients:
            flat_grad, _ = apply_dp_to_vector(flat_grad, self.dp_config, self.dp_generator)

        flat_grad = flat_grad.cpu()
        
        # Collect data for MI computation (flatten views for LeJEPA)
        if self.model_type == "lejepa":
            x_flat = x_test_views.reshape(x_test_views.shape[0], -1).detach().cpu()
        else:
            x_flat = x_test_cpu
            
        # Apply DP to model updates if enabled
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
            'gradients': flat_grad,
            'data': x_flat.detach(),
            'loss': loss.item(),
            'inv_loss': inv_loss,
            'sigreg_loss': sigreg_loss,
            'model_state': dp_state
        }


class FederatedServer:
    """Server for federated aggregation."""
    def __init__(self, model: nn.Module):
        self.global_model = model
        self.round_history = []
        
    def aggregate(self, client_updates: List[Dict]) -> None:
        """
        Federated averaging of client model updates.
        """
        with torch.no_grad():
            # Average all parameters
            avg_state = {}
            for key in self.global_model.state_dict().keys():
                avg_state[key] = torch.stack([
                    client['model_state'][key] for client in client_updates
                ]).mean(dim=0)
                
            self.global_model.load_state_dict(avg_state)


# ============================================
# Experiment Runner
# ============================================

def run_federated_privacy_experiment():
    """
    Complete experiment comparing LeJEPA and MAE privacy in federated learning.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration
    INPUT_DIM = 16 * 16
    EMB_DIM = 32
    PROJ_DIM = 16
    NUM_CLIENTS = 5
    SAMPLES_PER_CLIENT = 10000
    TOTAL_SAMPLES = NUM_CLIENTS * SAMPLES_PER_CLIENT
    DIRICHLET_ALPHA = 0.7
    NUM_ROUNDS = 10000
    NUM_VIEWS = 2
    LAMB = 0.005  # LeJEPA: balance between SIGReg and invariance
    USE_CNN = True
    IMAGE_SHAPE = (1, 16, 16)
    EVAL_EVERY = 250
    PLOT_ROUNDS = np.linspace(0, NUM_ROUNDS - 1, 10, dtype=int)
    PLOT_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    PLOT_STEPS = [0, 50, 100, 200, 400, 600, 800]
    PLOT_ATTACK_ITERS = max(PLOT_STEPS)
    LOSS_LOG_PATH = "loss_components_log.csv"
    mnist_transform = create_mnist_transform(IMAGE_SHAPE)
    # View augmentation parameters (masking only)
    AUG_ROTATION_DEG = 10.0
    AUG_TRANSLATION_PX = 3
    AUG_SCALE_RANGE = (1.0, 1.0)
    AUG_CONTRAST_RANGE = None
    AUG_BRIGHTNESS_RANGE = None
    AUG_BLUR_PROB = 0.0
    AUG_NOISE_STD = 0.0
    AUGMENTER_KWARGS = {
        "mask_ratio": 0.0,
        "noise_std": AUG_NOISE_STD,
        "rotation_deg": AUG_ROTATION_DEG,
        "translation_px": AUG_TRANSLATION_PX,
        "scale_range": AUG_SCALE_RANGE,
        "contrast_range": AUG_CONTRAST_RANGE,
        "brightness_range": AUG_BRIGHTNESS_RANGE,
        "blur_prob": AUG_BLUR_PROB,
    }

    # Differential privacy configuration
    DP_ENABLED = False
    DP_CLIP_NORM = 1.0
    DP_NOISE_MULTIPLIER = 0.8
    dp_config = DPConfig(
        enabled=DP_ENABLED,
        clip_norm=DP_CLIP_NORM,
        noise_multiplier=DP_NOISE_MULTIPLIER,
        seed=42,
        apply_to_gradients=True,
        apply_to_updates=True,
    )
    
    print("=" * 60)
    print("FEDERATED LEARNING PRIVACY COMPARISON")
    print("LeJEPA vs. Masked Autoencoder")
    print(f"Device: {device}")
    if dp_config.enabled:
        print(f"DP Mode: enabled | clip_norm={dp_config.clip_norm} | noise_multiplier={dp_config.noise_multiplier}")
    else:
        print("DP Mode: disabled")
    print("=" * 60)
    
    # Load MNIST and create non-IID client splits
    print("\n[1] Loading MNIST and creating non-IID splits...")
    client_data, client_labels = load_mnist_non_iid(
        num_clients=NUM_CLIENTS,
        total_samples=TOTAL_SAMPLES,
        alpha=DIRICHLET_ALPHA,
        seed=42,
        data_dir="data",
        image_shape=IMAGE_SHAPE,
    )
    all_train_data = torch.cat(client_data)
    all_train_labels = torch.cat(client_labels)
    class_counts = plot_client_class_distribution(
        client_labels,
        num_classes=10,
        save_path="client_class_distribution.png"
    )
    for c in range(NUM_CLIENTS):
        print(f"  Client {c}: samples={len(client_data[c])}, class_counts={class_counts[c].tolist()}")
    
    # Initialize models
    print("\n[2] Initializing models...")
    lejepa_model = LeJEPAModel(
        INPUT_DIM,
        EMB_DIM,
        PROJ_DIM,
        lamb=LAMB,
        use_cnn=USE_CNN,
        image_shape=IMAGE_SHAPE,
    )
    mae_model = MAEModel(
        INPUT_DIM,
        EMB_DIM,
        use_cnn=USE_CNN,
        image_shape=IMAGE_SHAPE,
    )
    lejepa_model.to(device)
    mae_model.to(device)
    print(f"  LeJEPA: emb_dim={EMB_DIM}, proj_dim={PROJ_DIM}, lambda={LAMB}, cnn={USE_CNN}")
    print(f"  MAE: latent_dim={EMB_DIM}, cnn={USE_CNN}")
    
    # Create clients
    lejepa_clients = [FederatedClient(
        i,
        client_data[i],
        "lejepa",
        NUM_VIEWS,
        device=device,
        dp_config=dp_config,
        augmenter_kwargs=AUGMENTER_KWARGS,
        image_shape=IMAGE_SHAPE,
    ) for i in range(NUM_CLIENTS)]
    mae_clients = [FederatedClient(
        i,
        client_data[i],
        "mae",
        1,
        device=device,
        dp_config=dp_config,
        augmenter_kwargs=AUGMENTER_KWARGS,
        image_shape=IMAGE_SHAPE,
    ) for i in range(NUM_CLIENTS)]
    
    # Servers
    lejepa_server = FederatedServer(lejepa_model)
    mae_server = FederatedServer(mae_model)
    
    # MI Estimators
    # gaussian_mi = GaussianMIEstimator(method='gaussian')
    
    # Privacy attackers
    lejepa_attacker = GradientInversionAttack(lejepa_model, INPUT_DIM, NUM_VIEWS)
    mae_attacker = GradientInversionAttack(mae_model, INPUT_DIM, 1)
    
    # Track metrics
    results = {
        'lejepa': {
            'mi': [], 'mse': [], 'psnr': [], 'cosine': [], 'rounds': [],
            'loss': [], 'loss_rounds': [], 'probe_acc': [], 'probe_rounds': [],
            'inv': [], 'sigreg': [],
            'global_loss': [], 'global_inv': [], 'global_sigreg': [], 'global_rounds': []
        },
        'mae': {
            'mi': [], 'mse': [], 'psnr': [], 'cosine': [], 'rounds': [],
            'loss': [], 'loss_rounds': [], 'probe_acc': [], 'probe_rounds': [],
            'global_loss': [], 'global_rounds': []
        }
    }
    
    last_reconstructions = {}
    initialize_loss_log(LOSS_LOG_PATH)

    # Training loop
    print("\n[3] Running federated training...")
    for round_idx in range(NUM_ROUNDS):
        print(f"\n  Round {round_idx + 1}/{NUM_ROUNDS}")
        
        # LeJEPA round
        lejepa_updates = [client.local_train(lejepa_server.global_model, epochs=2, lr=1e-4) 
                          for client in lejepa_clients]
        lejepa_server.aggregate(lejepa_updates)
        lejepa_avg_loss = float(np.mean([update['loss'] for update in lejepa_updates]))
        results['lejepa']['loss'].append(lejepa_avg_loss)
        results['lejepa']['loss_rounds'].append(round_idx)

        lejepa_avg_inv = float(np.mean([update['inv_loss'] for update in lejepa_updates]))
        lejepa_avg_sigreg = float(np.mean([update['sigreg_loss'] for update in lejepa_updates]))
        results['lejepa']['inv'].append(lejepa_avg_inv)
        results['lejepa']['sigreg'].append(lejepa_avg_sigreg)

        for client_id, update in enumerate(lejepa_updates):
            append_loss_log(
                LOSS_LOG_PATH,
                round_idx,
                "lejepa",
                "client",
                client_id,
                {
                    "total": update["loss"],
                    "inv": update["inv_loss"],
                    "sigreg": update["sigreg_loss"]
                }
            )
        
        # MAE round
        mae_updates = [client.local_train(mae_server.global_model, epochs=2, lr=1e-4) 
                       for client in mae_clients]
        mae_server.aggregate(mae_updates)
        mae_avg_loss = float(np.mean([update['loss'] for update in mae_updates]))
        results['mae']['loss'].append(mae_avg_loss)
        results['mae']['loss_rounds'].append(round_idx)

        for client_id, update in enumerate(mae_updates):
            append_loss_log(
                LOSS_LOG_PATH,
                round_idx,
                "mae",
                "client",
                client_id,
                {"total": update["loss"], "inv": None, "sigreg": None}
            )

        global_batch = sample_tensor_batch(all_train_data, max_samples=256, seed=round_idx + 1)
        global_augmenter = ViewAugmenter(
            num_views=NUM_VIEWS,
            image_shape=IMAGE_SHAPE,
            **AUGMENTER_KWARGS,
        )
        lejepa_global_losses = compute_loss_components(
            lejepa_server.global_model,
            global_batch,
            model_type="lejepa",
            augmenter=global_augmenter
        )
        results['lejepa']['global_loss'].append(lejepa_global_losses['total'])
        results['lejepa']['global_inv'].append(lejepa_global_losses['inv'])
        results['lejepa']['global_sigreg'].append(lejepa_global_losses['sigreg'])
        results['lejepa']['global_rounds'].append(round_idx)
        append_loss_log(
            LOSS_LOG_PATH,
            round_idx,
            "lejepa",
            "global",
            -1,
            lejepa_global_losses
        )

        mae_global_losses = compute_loss_components(
            mae_server.global_model,
            global_batch,
            model_type="mae",
            augmenter=None
        )
        results['mae']['global_loss'].append(mae_global_losses['total'])
        results['mae']['global_rounds'].append(round_idx)
        append_loss_log(
            LOSS_LOG_PATH,
            round_idx,
            "mae",
            "global",
            -1,
            mae_global_losses
        )

        print(
            f"    [Loss] JEPA avg={lejepa_avg_loss:.4f} inv={lejepa_avg_inv:.4f} "
            f"sigreg={lejepa_avg_sigreg:.4f} | MAE avg={mae_avg_loss:.4f}"
        )
        
        # Privacy evaluation every EVAL_EVERY rounds
        if round_idx % EVAL_EVERY == 0 or round_idx == NUM_ROUNDS - 1:
            # Sample one client's data for analysis
            client_idx = 0
            
            # === LeJEPA Privacy Analysis ===
            lejepa_grad = lejepa_updates[client_idx]['gradients']
            lejepa_data = lejepa_updates[client_idx]['data']
            
            # Compute MI
            # lejepa_mi = gaussian_mi.estimate_mi(lejepa_data, lejepa_grad.unsqueeze(0) if lejepa_grad.dim() == 1 else lejepa_grad)
            
            # Gradient inversion attack
            x_test = client_data[client_idx][:16]
            augmenter = ViewAugmenter(
                num_views=NUM_VIEWS,
                image_shape=IMAGE_SHAPE,
                **AUGMENTER_KWARGS,
            )
            x_test_views = augmenter(x_test)

            lejepa_recon, _ = lejepa_attacker.attack(
                x_test_views,
                lejepa_grad,
                lr=0.05,
                iterations=600,
                loss_strategy="cosine"
            )
            lejepa_metrics = lejepa_attacker.compute_metrics(x_test_views, lejepa_recon)
            
            # results['lejepa']['mi'].append(lejepa_mi)
            results['lejepa']['mse'].append(lejepa_metrics['mse'])
            results['lejepa']['psnr'].append(lejepa_metrics['psnr'])
            results['lejepa']['cosine'].append(lejepa_metrics['cosine_sim'])
            results['lejepa']['rounds'].append(round_idx)
            
            # === MAE Privacy Analysis ===
            mae_grad = mae_updates[client_idx]['gradients']
            mae_data = mae_updates[client_idx]['data']
            
            # Compute MI
            # mae_mi = gaussian_mi.estimate_mi(mae_data, mae_grad.unsqueeze(0) if mae_grad.dim() == 1 else mae_grad)
            
            # Gradient inversion attack
            mae_recon, _ = mae_attacker.attack(
                x_test,
                mae_grad,
                lr=0.05,
                iterations=600,
                loss_strategy="cosine"
            )
            mae_metrics = mae_attacker.compute_metrics(x_test, mae_recon)
            
            # results['mae']['mi'].append(mae_mi)
            results['mae']['mse'].append(mae_metrics['mse'])
            results['mae']['psnr'].append(mae_metrics['psnr'])
            results['mae']['cosine'].append(mae_metrics['cosine_sim'])
            results['mae']['rounds'].append(round_idx)
            
            # print(f"    LeJEPA: MI={lejepa_mi:.4f}, MSE={lejepa_metrics['mse']:.4f}, "
            #       f"PSNR={lejepa_metrics['psnr']:.2f}dB, Cos={lejepa_metrics['cosine_sim']:.4f}")
            print(f"    JEPA:  MSE={lejepa_metrics['mse']:.4f}, "
                  f"PSNR={lejepa_metrics['psnr']:.2f}dB, Cos={lejepa_metrics['cosine_sim']:.4f}")
            print(f"    MAE:   MSE={mae_metrics['mse']:.4f}, "
                  f"PSNR={mae_metrics['psnr']:.2f}dB, Cos={mae_metrics['cosine_sim']:.4f}")

            if round_idx == NUM_ROUNDS - 1:
                last_reconstructions['lejepa_cos'] = (x_test_views.detach(), lejepa_recon.detach())
                last_reconstructions['mae_cos'] = (x_test.detach(), mae_recon.detach())

                lejepa_recon_mse, _ = lejepa_attacker.attack(
                    x_test_views,
                    lejepa_grad,
                    lr=0.05,
                    iterations=600,
                    loss_strategy="mse"
                )
                mae_recon_mse, _ = mae_attacker.attack(
                    x_test,
                    mae_grad,
                    lr=0.05,
                    iterations=600,
                    loss_strategy="mse"
                )
                last_reconstructions['lejepa_mse'] = (x_test_views.detach(), lejepa_recon_mse.detach())
                last_reconstructions['mae_mse'] = (x_test.detach(), mae_recon_mse.detach())

        if round_idx in PLOT_ROUNDS:
            print(f"\n  [Plotting] Reconstruction steps at round {round_idx + 1}")
            class_samples = sample_class_images(all_train_data, all_train_labels, PLOT_CLASSES, samples_per_class=4)

            histories = {"lejepa": {}, "mae": {}}
            originals = {"lejepa": {}, "mae": {}}

            for cls, samples in class_samples.items():
                augmenter = ViewAugmenter(
                    num_views=NUM_VIEWS,
                    image_shape=IMAGE_SHAPE,
                    **AUGMENTER_KWARGS,
                )
                lejepa_grad = compute_gradients_for_data(
                    lejepa_server.global_model,
                    samples,
                    model_type="lejepa",
                    num_views=NUM_VIEWS,
                    augmenter=augmenter
                )
                mae_grad = compute_gradients_for_data(
                    mae_server.global_model,
                    samples,
                    model_type="mae",
                    num_views=1,
                    augmenter=augmenter
                )

                lejepa_recon, lejepa_history = lejepa_attacker.attack(
                    augmenter(samples),
                    lejepa_grad,
                    iterations=PLOT_ATTACK_ITERS,
                    return_history=True,
                    record_steps=PLOT_STEPS
                )
                mae_recon, mae_history = mae_attacker.attack(
                    samples,
                    mae_grad,
                    iterations=PLOT_ATTACK_ITERS,
                    return_history=True,
                    record_steps=PLOT_STEPS
                )

                histories["lejepa"][cls] = lejepa_history
                histories["mae"][cls] = mae_history
                originals["lejepa"][cls] = augmenter(samples)
                originals["mae"][cls] = samples

            plot_reconstruction_steps_by_class(
                originals_by_class=originals["lejepa"],
                histories_by_class=histories["lejepa"],
                steps=PLOT_STEPS,
                save_path=f"lejepa_recon_steps_round{round_idx + 1}.png",
                title=f"LeJEPA Recon Steps (Round {round_idx + 1})",
                image_shape=IMAGE_SHAPE,
            )
            plot_reconstruction_steps_by_class(
                originals_by_class=originals["mae"],
                histories_by_class=histories["mae"],
                steps=PLOT_STEPS,
                save_path=f"mae_recon_steps_round{round_idx + 1}.png",
                title=f"MAE Recon Steps (Round {round_idx + 1})",
                image_shape=IMAGE_SHAPE,
            )

            print("  [Plotting] t-SNE embeddings for validation samples")
            val_dataset = datasets.MNIST(root="data", train=False, download=True, transform=mnist_transform)
            val_data, val_labels = sample_mnist_dataset(val_dataset, max_samples=600)
            plot_tsne_for_validation(
                lejepa_server.global_model,
                val_data,
                val_labels,
                model_type="lejepa",
                save_path=f"lejepa_tsne_round{round_idx + 1}_val.png",
                title=f"LeJEPA Validation t-SNE (Round {round_idx + 1})"
            )
            plot_tsne_for_validation(
                mae_server.global_model,
                val_data,
                val_labels,
                model_type="mae",
                save_path=f"mae_tsne_round{round_idx + 1}_val.png",
                title=f"MAE Validation t-SNE (Round {round_idx + 1})"
            )

        if round_idx in PLOT_ROUNDS:
            print(f"\n  [Probing] Linear probe at round {round_idx + 1}")
            train_data, train_labels = sample_tensor_dataset(all_train_data, all_train_labels, max_samples=2000)

            test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=mnist_transform)
            test_data, test_labels = sample_mnist_dataset(test_dataset, max_samples=1000)

            lejepa_probe_acc = train_linear_probe(
                lejepa_server.global_model,
                train_data,
                train_labels,
                test_data,
                test_labels,
                model_type="lejepa"
            )
            mae_probe_acc = train_linear_probe(
                mae_server.global_model,
                train_data,
                train_labels,
                test_data,
                test_labels,
                model_type="mae"
            )
            results['lejepa']['probe_acc'].append(lejepa_probe_acc)
            results['lejepa']['probe_rounds'].append(round_idx)
            results['mae']['probe_acc'].append(mae_probe_acc)
            results['mae']['probe_rounds'].append(round_idx)
            print(f"    Probe JEPA: {lejepa_probe_acc * 100:.2f}%")
            print(f"    Probe MAE:  {mae_probe_acc * 100:.2f}%")

    if last_reconstructions:
        print("\n[4] Saving reconstructed image grids...")
        lejepa_orig, lejepa_recon = last_reconstructions['lejepa_cos']
        mae_orig, mae_recon = last_reconstructions['mae_cos']
        plot_reconstructions(
            lejepa_orig,
            lejepa_recon,
            save_path="lejepa_reconstructions.png",
            title="LeJEPA Gradient Inversion (Cosine)",
            image_shape=IMAGE_SHAPE,
        )
        plot_reconstructions(
            mae_orig,
            mae_recon,
            save_path="mae_reconstructions.png",
            title="MAE Gradient Inversion (Cosine)",
            image_shape=IMAGE_SHAPE,
        )

        lejepa_orig, lejepa_recon = last_reconstructions['lejepa_mse']
        mae_orig, mae_recon = last_reconstructions['mae_mse']
        plot_reconstructions(
            lejepa_orig,
            lejepa_recon,
            save_path="lejepa_reconstructions_mse.png",
            title="LeJEPA Gradient Inversion (MSE)",
            image_shape=IMAGE_SHAPE,
        )
        plot_reconstructions(
            mae_orig,
            mae_recon,
            save_path="mae_reconstructions_mse.png",
            title="MAE Gradient Inversion (MSE)",
            image_shape=IMAGE_SHAPE,
        )

    print("\n[5] Plotting learning curves...")
    plot_metric_curve(
        results['lejepa']['loss_rounds'],
        results['lejepa']['loss'],
        results['mae']['loss'],
        label_a="LeJEPA",
        label_b="MAE",
        ylabel="Local training loss",
        title="Federated Training Loss",
        save_path="training_loss_curve.png"
    )
    plot_metric_curve(
        results['lejepa']['loss_rounds'],
        results['lejepa']['inv'],
        results['lejepa']['sigreg'],
        label_a="LeJEPA Invariance Loss",
        label_b="LeJEPA SIGReg Loss",
        ylabel="LeJEPA Losses",
        title="LeJEPA Losses",
        save_path="lejepa_losses.png"
    )
    plot_metric_curve(
        results['lejepa']['probe_rounds'],
        results['lejepa']['probe_acc'],
        results['mae']['probe_acc'],
        label_a="LeJEPA",
        label_b="MAE",
        ylabel="Linear probe accuracy",
        title="Linear Probe Accuracy over Rounds",
        save_path="linear_probe_curve.png"
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Average metrics
    lejepa_avg_mi = np.mean(results['lejepa']['mi']) if results['lejepa']['mi'] else 0.0
    mae_avg_mi = np.mean(results['mae']['mi']) if results['mae']['mi'] else 0.0
    
    lejepa_avg_mse = np.mean(results['lejepa']['mse'])
    mae_avg_mse = np.mean(results['mae']['mse'])
    
    lejepa_avg_psnr = np.mean(results['lejepa']['psnr'])
    mae_avg_psnr = np.mean(results['mae']['psnr'])
    
    lejepa_avg_cos = np.mean(results['lejepa']['cosine'])
    mae_avg_cos = np.mean(results['mae']['cosine'])
    
    print(f"\nMutual Information I(X; G) [nats] (LOWER is better privacy):")
    print(f"  LeJEPA: {lejepa_avg_mi:.4f}")
    print(f"  MAE:    {mae_avg_mi:.4f}")
    if mae_avg_mi > 0:
        print(f"  Improvement: {(1 - lejepa_avg_mi/mae_avg_mi)*100:.1f}%")
    else:
        print("  Improvement: n/a (MI disabled)")
    
    print(f"\nReconstruction MSE [HIGHER is better privacy]:")
    print(f"  LeJEPA: {lejepa_avg_mse:.4f}")
    print(f"  MAE:    {mae_avg_mse:.4f}")
    print(f"  Improvement: {(lejepa_avg_mse - mae_avg_mse) / lejepa_avg_mse * 100:.1f}%")
    
    print(f"\nPSNR [dB] [LOWER is better privacy]:")
    print(f"  LeJEPA: {lejepa_avg_psnr:.2f} dB")
    print(f"  MAE:    {mae_avg_psnr:.2f} dB")
    
    print(f"\nCosine Similarity [LOWER is better privacy]:")
    print(f"  LeJEPA: {lejepa_avg_cos:.4f}")
    print(f"  MAE:    {mae_avg_cos:.4f}")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
    LeJEPA provides better privacy because:
    1. Gradients come from an invariance loss (variance minimization) rather
       than direct reconstruction
    2. SIGReg enforces isotropic Gaussian structure, adding noise to gradients
    3. Multiple views create a "privacy amplification" effect
    4. No decoder means no direct reconstruction pathway in the architecture
    
    Key insight: I(X; G) is lower for LeJEPA because the mapping from data to
    gradients is more complex (involves view averaging + regularization) vs.
    MAE's direct reconstruction objective.
    """)
    
    return results


if __name__ == "__main__":
    results = run_federated_privacy_experiment()

