"""Data augmentation utilities for MNIST experiments."""

from __future__ import annotations

from typing import Tuple

import torch
from torchvision.transforms import v2

from .data import normalize_mnist


class ViewAugmenter:
    """Create multiple augmented views of the same sample with diverse transforms."""

    def __init__(
        self,
        num_views: int = 2,
        mask_ratio: float = 0.4,
        noise_std: float = 0.1,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
        device: str = "cuda",
        normalize_mean: float = 0.1307,
        normalize_std: float = 0.3081,
        mask_mode: str = "pixel",
        patch_size: int = 4,
        deterministic: bool = False,
        base_seed: int = 42,
        **kwargs,
    ):
        self.num_views = num_views
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std
        self.image_shape = image_shape
        self.device = device
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.mask_mode = mask_mode
        self.patch_size = patch_size
        self.deterministic = deterministic
        self.base_seed = base_seed
        self._call_count = 0

        self.spatial_transform = v2.RandomAffine(
            degrees=kwargs.get("rotation_deg", 20.0),
            translate=(
                kwargs.get("translation_px", 3) / image_shape[1],
                kwargs.get("translation_px", 3) / image_shape[2],
            ),
            scale=kwargs.get("scale_range", (0.9, 1.1)),
            interpolation=v2.InterpolationMode.BILINEAR,
        )

        self.color_transform = v2.Compose(
            [
                v2.ColorJitter(
                    brightness=kwargs.get("brightness_range", (0.85, 1.15)),
                    contrast=kwargs.get("contrast_range", (0.8, 1.2)),
                    saturation=0.0,
                    hue=0.0,
                ),
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                    p=kwargs.get("blur_prob", 0.25),
                ),
            ]
        )

        self.perspective = v2.RandomApply(
            [v2.RandomPerspective(distortion_scale=0.3, p=1.0)],
            p=kwargs.get("perspective_prob", 0.0),
        )
        solarize_threshold = kwargs.get("solarize_threshold", 128)
        if isinstance(solarize_threshold, (int, float)) and solarize_threshold > 1.0:
            solarize_threshold = solarize_threshold / 255.0
        self.solarize = v2.RandomApply(
            [v2.RandomSolarize(threshold=solarize_threshold)],
            p=kwargs.get("solarize_prob", 0.0),
        )

    def _mask_pixels(self, view: torch.Tensor) -> torch.Tensor:
        mask = torch.rand_like(view) > self.mask_ratio
        return view * mask.float()

    def _mask_patches(self, view: torch.Tensor) -> torch.Tensor:
        if self.mask_ratio <= 0:
            return view
        b, _ = view.shape
        channels, height, width = self.image_shape
        patch = self.patch_size
        if height % patch != 0 or width % patch != 0:
            raise ValueError("image_shape height/width must be divisible by patch_size")
        view_img = view.view(b, channels, height, width)
        grid_h = height // patch
        grid_w = width // patch
        mask = torch.rand((b, grid_h, grid_w), device=view.device) > self.mask_ratio
        mask = mask.repeat_interleave(patch, dim=1).repeat_interleave(patch, dim=2)
        mask = mask.unsqueeze(1)
        masked = view_img * mask.float()
        return masked.view(b, -1)

    def _apply_mask(self, view: torch.Tensor) -> torch.Tensor:
        if self.mask_mode == "none":
            return view
        if self.mask_mode == "patch":
            return self._mask_patches(view)
        return self._mask_pixels(view)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_dim) - batch of samples in [0, 1]
        Returns: (B, num_views, input_dim) normalized
        """
        b, _ = x.shape
        channels, height, width = self.image_shape

        x_img = x.view(b, channels, height, width).to(self.device)

        views = []
        for _ in range(self.num_views):
            if self.deterministic:
                seed = self.base_seed + self._call_count
                self._call_count += 1
                with torch.random.fork_rng(devices=[x_img.device] if x_img.is_cuda else []):
                    torch.manual_seed(seed)
                    view = self._apply_transforms(x_img)
            else:
                view = self._apply_transforms(x_img)

            views.append(view.unsqueeze(1))

        return torch.cat(views, dim=1)

    def _apply_transforms(self, x_img: torch.Tensor) -> torch.Tensor:
        view = x_img.clone()

        view = self.spatial_transform(view)
        view = self.color_transform(view)
        view = self.perspective(view)
        view = self.solarize(view)

        b = view.shape[0]
        view = view.view(b, -1)
        view = self._apply_mask(view)

        if self.noise_std > 0:
            view = view + torch.randn_like(view) * self.noise_std

        view = normalize_mnist(view, self.normalize_mean, self.normalize_std)
        return view


class IdentityAugmenter:
    def __init__(self, normalize_mean: float, normalize_std: float):
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return normalize_mnist(x, self.normalize_mean, self.normalize_std)


class IdentityViewAugmenter:
    def __init__(self, num_views: int, normalize_mean: float, normalize_std: float):
        self.num_views = num_views
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        normalized = normalize_mnist(x, self.normalize_mean, self.normalize_std)
        return normalized.unsqueeze(1).repeat(1, self.num_views, 1)
