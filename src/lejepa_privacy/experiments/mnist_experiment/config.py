"""Configuration objects and loader for the MNIST experiment."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class DPConfig:
    enabled: bool = False
    clip_norm: float = 1.0
    noise_multiplier: float = 0.8
    seed: int = 42
    apply_to_gradients: bool = True
    apply_to_updates: bool = True


@dataclass
class ExperimentConfig:
    seed: int = 42
    deterministic: bool = True
    device: str = "auto"

    input_dim: int = 32 * 32
    emb_dim: int = 32
    proj_dim: int = 16
    num_clients: int = 5
    samples_per_client: int = 15000
    dirichlet_alpha: float = 10.0
    num_rounds: int = 1000
    clients_per_round: int | None = None
    num_views: int = 4
    lamb: float = 0.005
    use_cnn: bool = False
    use_vit: bool = True
    image_shape: Tuple[int, int, int] = (1, 32, 32)

    eval_every: int = 250
    plot_rounds: List[int] = field(default_factory=list)
    plot_classes: List[int] = field(default_factory=lambda: [0, 1, 2])
    plot_steps: List[int] = field(default_factory=lambda: [0, 50, 100, 200, 400])

    batch_size: int = 128
    local_epochs: int = 1
    max_batches_per_epoch: int | None = None
    optimizer: str = "sgd"
    learning_rate: float = 1e-3
    data_loader_num_workers: int = 0
    data_loader_pin_memory: bool = True
    global_batch_size: int = 256

    normalize_mean: float = 0.1307
    normalize_std: float = 0.3081

    augmenter_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "mask_ratio": 0.0,
            "noise_std": 0.05,
            "rotation_deg": 25.0,
            "translation_px": 4,
            "scale_range": (0.8, 1.2),
            "contrast_range": (0.7, 1.3),
            "brightness_range": (0.7, 1.3),
            "blur_prob": 0.35,
            "perspective_prob": 0.25,
            "solarize_prob": 0.2,
            "solarize_threshold": 128,
            "mask_mode": "pixel",
            "patch_size": 4,
        }
    )
    mae_augmenter_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "mask_ratio": 0.0,
            "noise_std": 0.0,
            "rotation_deg": 0.0,
            "translation_px": 0,
            "scale_range": (1.0, 1.0),
            "contrast_range": (1.0, 1.0),
            "brightness_range": (1.0, 1.0),
            "blur_prob": 0.0,
            "perspective_prob": 0.0,
            "solarize_prob": 0.0,
            "solarize_threshold": 0,
            "mask_mode": "patch",
            "patch_size": 4,
        }
    )

    dp_config: DPConfig = field(default_factory=DPConfig)

    align_augmentations: bool = True
    federated_strategy: str = "updates"
    attack_on: str = "updates"
    attack_loss_strategies: List[str] = field(default_factory=lambda: ["cosine", "mse"])
    attack_eval_clients: int = 1
    attack_eval_batches: int = 1
    attack_iterations: int = 200
    attack_plot_iterations: int = 200
    attack_success_mse_threshold: float = 0.05
    attack_use_raw_data: bool = True
    attack_deterministic_augment: bool = True
    attack_seed: int = 123

    val_tsne_samples: int = 600
    probe_train_samples: int = 2000
    probe_test_samples: int = 1000

    output_dir: str = "results"
    checkpoint_every: int = 250
    resume_from: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["image_shape"] = list(self.image_shape)
        return data


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None) -> ExperimentConfig:
    config = ExperimentConfig()
    if not config_path:
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    merged = _merge_dict(config.to_dict(), payload)
    dp_config = DPConfig(**merged.get("dp_config", {}))
    merged["dp_config"] = dp_config

    if "image_shape" in merged:
        image_shape = tuple(merged["image_shape"])
        merged["image_shape"] = image_shape

    return ExperimentConfig(**merged)
