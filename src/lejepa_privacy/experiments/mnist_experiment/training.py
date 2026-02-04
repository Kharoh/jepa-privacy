"""Federated training loop and evaluation for MNIST privacy experiment."""

from __future__ import annotations

import copy
import csv
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from .augment import IdentityAugmenter, ViewAugmenter
from .config import ExperimentConfig
from .data import (
    create_mnist_transform,
    denormalize_mnist,
    load_mnist_non_iid,
    normalize_mnist,
    sample_class_images,
    sample_mnist_dataset,
    sample_tensor_batch,
    sample_tensor_dataset,
)
from .metrics import bootstrap_ci, permutation_test, summarize_metric
from .models import LeJEPAModel, MAEModel, MAEViT
from .plotting import (
    plot_client_class_distribution,
    plot_metric_curve,
    plot_scalar_curve,
    plot_tradeoff_curve,
    plot_reconstruction_steps_by_class,
    plot_reconstructions,
    plot_tsne_latents,
)
from .privacy import GradientInversionAttack, UpdateInversionAttack, apply_dp_to_tensors, apply_dp_to_vector
from .utils import checkpoint_path


def extract_latents_for_tsne(
    model: nn.Module,
    data: torch.Tensor,
    model_type: str,
    image_shape: Tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Extract latent vectors for t-SNE visualization."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        if model_type == "lejepa":
            if data.dim() == 2:
                data = data.unsqueeze(1)
            elif data.dim() == 3 and data.shape[1] != 1:
                data = data[:, :1, :]
            emb, _ = model.encoder(data.to(device))
            latents = emb.mean(dim=1)
        else:
            input_tensor = data.to(device)
            if image_shape is not None:
                c, h, w = image_shape
            else:
                c = 1
                h = w = (
                    int(np.sqrt(input_tensor.shape[-1]))
                    if input_tensor.dim() == 2
                    else int(np.sqrt(input_tensor.shape[-2]))
                )
            if input_tensor.dim() == 2:
                input_tensor = input_tensor.view(-1, c, h, w)
            elif input_tensor.dim() == 3:
                if input_tensor.shape[1] != c:
                    input_tensor = input_tensor.view(input_tensor.shape[0], c, h, w)
            if hasattr(model, "encode"):
                latents = model.encode(input_tensor)
            else:
                latents = model.encoder(input_tensor)
    return latents.detach().cpu()


def plot_tsne_for_validation(
    model: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    model_type: str,
    save_path: str,
    title: str,
    image_shape: Tuple[int, int, int] | None = None,
) -> None:
    latents = extract_latents_for_tsne(model, data, model_type, image_shape=image_shape)
    plot_tsne_latents(latents, labels, save_path=save_path, title=title)


def extract_features(
    model: nn.Module,
    data: torch.Tensor,
    model_type: str,
    batch_size: int = 256,
    image_shape: Tuple[int, int, int] = (1, 28, 28),
) -> torch.Tensor:
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
                if batch.dim() == 2:
                    expected_dim = image_shape[0] * image_shape[1] * image_shape[2]
                    if batch.shape[1] != expected_dim:
                        raise ValueError(
                            f"Expected flattened input dim {expected_dim}, got {batch.shape[1]}"
                        )
                    batch = batch.view(-1, *image_shape)
                if hasattr(model, "encode"):
                    feats = model.encode(batch)
                else:
                    feats = model.encoder(batch)
            features.append(feats.detach().cpu())

    return torch.cat(features, dim=0)


def train_linear_probe(
    model: nn.Module,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    model_type: str,
    epochs: int = 20,
    lr: float = 1e-2,
    batch_size: int = 256,
    image_shape: Tuple[int, int, int] = (1, 28, 28),
) -> float:
    """Train a detached linear probe with Batch Normalization."""

    train_features = extract_features(model, train_data, model_type, image_shape=image_shape)
    test_features = extract_features(model, test_data, model_type, image_shape=image_shape)

    device = next(model.parameters()).device
    feat_dim = train_features.shape[1]

    probe = nn.Sequential(
        nn.BatchNorm1d(feat_dim, affine=False),
        nn.Linear(feat_dim, 10),
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

    probe.eval()
    with torch.no_grad():
        test_feats = test_features.to(device)
        test_labels = test_labels.to(device)

        logits = probe(test_feats)
        preds = logits.argmax(dim=1)
        acc = (preds == test_labels).float().mean().item()

    return acc


def compute_loss_components(
    model: nn.Module,
    batch: torch.Tensor,
    model_type: str,
    augmenter: ViewAugmenter | IdentityAugmenter | None = None,
) -> Dict[str, float]:
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
            "sigreg": float(loss_dict["sigreg"].item()),
        }
    if augmenter is not None:
        batch = augmenter(batch)
    loss = model.reconstruction_loss(batch)
    return {"total": float(loss.item()), "inv": None, "sigreg": None}


def compute_gradients_for_data(
    model: nn.Module,
    x: torch.Tensor,
    model_type: str,
    num_views: int,
    augmenter: ViewAugmenter | IdentityAugmenter,
) -> torch.Tensor:
    """Compute flattened gradients for a batch without updating model weights."""
    device = next(model.parameters()).device
    x = x.to(device)
    model.zero_grad(set_to_none=True)

    if model_type == "lejepa":
        if x.dim() == 2:
            x = augmenter(x)
        loss = model.compute_loss(x)["total"]
    else:
        if x.dim() == 2:
            x = augmenter(x)
        loss = model.reconstruction_loss(x)

    loss.backward()
    grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    return grads.detach().cpu()


def initialize_loss_log(log_path: Path) -> None:
    """Initialize CSV log for loss components."""
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["round", "model", "scope", "client_id", "loss_total", "loss_inv", "loss_sigreg"]
        )


def append_loss_log(
    log_path: Path,
    round_idx: int,
    model: str,
    scope: str,
    client_id: int,
    loss_components: Dict[str, float],
) -> None:
    """Append a loss log row to CSV."""
    with log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                round_idx,
                model,
                scope,
                client_id,
                loss_components.get("total"),
                loss_components.get("inv"),
                loss_components.get("sigreg"),
            ]
        )


def initialize_attack_class_log(log_path: Path) -> None:
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["round", "model", "strategy", "class", "mse", "psnr", "cosine", "success"])


def append_attack_class_log(
    log_path: Path,
    round_idx: int,
    model: str,
    strategy: str,
    class_id: int,
    metrics: Dict[str, float],
    success: float,
) -> None:
    with log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                round_idx,
                model,
                strategy,
                class_id,
                metrics.get("mse"),
                metrics.get("psnr"),
                metrics.get("cosine_sim"),
                success,
            ]
        )


class FederatedClient:
    """Client for federated learning with gradient extraction and MI tracking."""

    def __init__(
        self,
        client_id: int,
        data: torch.Tensor,
        labels: torch.Tensor | None = None,
        model_type: str = "lejepa",
        num_views: int = 2,
        device: torch.device = torch.device("cpu"),
        dp_config=None,
        augmenter: ViewAugmenter | IdentityAugmenter | None = None,
    ):
        self.client_id = client_id
        self.data = data
        self.labels = labels
        self.model_type = model_type
        self.num_views = num_views
        self.augmenter = augmenter
        self.device = device
        self.dp_config = dp_config
        gen_device = "cuda" if device.type == "cuda" else "cpu"
        self.dp_generator = torch.Generator(device=gen_device)
        seed = dp_config.seed if dp_config is not None else 42
        self.dp_generator.manual_seed(seed + client_id)

    def local_train(
        self,
        global_model: nn.Module,
        epochs: int,
        lr: float,
        batch_size: int,
        max_batches: int | None,
        optimizer_name: str = "sgd",
    ) -> Dict:
        """
        Train locally and return gradients for privacy analysis.
        Gradients correspond to the last local update batch.
        """
        device = self.device
        local_model = copy.deepcopy(global_model)
        local_model.to(device)
        optimizer_name = optimizer_name.lower().strip()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        if self.labels is not None:
            dataset = TensorDataset(self.data, self.labels)
        else:
            dataset = TensorDataset(self.data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        last_batch = None
        last_labels = None
        last_loss = None
        last_grad = None
        last_input = None
        last_inv = None
        last_sigreg = None

        for _ in range(epochs):
            for batch_idx, batch in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                if self.labels is not None:
                    x_batch, y_batch = batch
                    last_labels = y_batch
                else:
                    x_batch = batch[0]
                    last_labels = None

                last_batch = x_batch
                x_batch = x_batch.to(device)

                optimizer.zero_grad()

                if self.model_type == "lejepa":
                    if self.augmenter is None:
                        raise ValueError("augmenter is required for LeJEPA training")
                    x_views = self.augmenter(x_batch)
                    loss_dict = local_model.compute_loss(x_views)
                    loss = loss_dict["total"]
                    last_inv = float(loss_dict["inv"].item())
                    last_sigreg = float(loss_dict["sigreg"].item())
                    last_input = x_views.detach()
                else:
                    if self.augmenter is not None:
                        x_aug = self.augmenter(x_batch)
                    else:
                        x_aug = x_batch
                    if x_aug.dim() == 3:
                        x_aug = x_aug[:, 0, :]
                    loss = local_model.reconstruction_loss(x_aug)
                    last_input = x_aug.detach()
                    last_inv = None
                    last_sigreg = None

                loss.backward()

                flat_grad = torch.cat(
                    [p.grad.flatten() for p in local_model.parameters() if p.grad is not None]
                ).detach()

                if self.dp_config is not None and self.dp_config.enabled and self.dp_config.apply_to_gradients:
                    flat_grad, _ = apply_dp_to_vector(flat_grad, self.dp_config, self.dp_generator)

                optimizer.step()

                last_loss = loss
                last_grad = flat_grad

        if last_batch is None or last_grad is None:
            raise RuntimeError("No batches processed during local training")

        last_batch_cpu = last_batch.detach().cpu()
        last_labels_cpu = last_labels.detach().cpu() if last_labels is not None else None

        if self.dp_config is not None and self.dp_config.enabled and self.dp_config.apply_to_updates:
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

        update_vector = torch.cat(
            [
                (local_param.detach() - global_param.detach()).flatten().cpu()
                for (_, local_param), (_, global_param) in zip(
                    local_model.named_parameters(), global_model.named_parameters()
                )
            ]
        )

        return {
            "gradients": last_grad.cpu(),
            "data": last_input.detach().cpu() if last_input is not None else last_batch_cpu,
            "raw_data": last_batch_cpu,
            "raw_labels": last_labels_cpu,
            "loss": float(last_loss.item() if last_loss is not None else 0.0),
            "inv_loss": last_inv,
            "sigreg_loss": last_sigreg,
            "model_state": dp_state,
            "update_vector": update_vector,
            "num_samples": len(self.data),
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
            avg_state = {}
            for key in self.global_model.state_dict().keys():
                stacked = []
                for client in client_updates:
                    tensor = client["model_state"][key]
                    if not tensor.is_floating_point():
                        tensor = tensor.to(torch.float32)
                    stacked.append(tensor)
                stacked = torch.stack(stacked)
                avg = stacked.mean(dim=0)
                ref = client_updates[0]["model_state"][key]
                if not ref.is_floating_point():
                    avg = avg.round().to(ref.dtype)
                avg_state[key] = avg

            self.global_model.load_state_dict(avg_state)


def _build_results_dict() -> Dict[str, Dict[str, List[float]]]:
    return {
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
            "global_loss": [],
            "global_inv": [],
            "global_sigreg": [],
            "global_rounds": [],
            "baseline_random_mse": [],
            "baseline_random_psnr": [],
            "baseline_random_cosine": [],
            "baseline_mean_mse": [],
            "baseline_mean_psnr": [],
            "baseline_mean_cosine": [],
            "attack": {},
            "attack_success": {},
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
            "global_loss": [],
            "global_rounds": [],
            "baseline_random_mse": [],
            "baseline_random_psnr": [],
            "baseline_random_cosine": [],
            "baseline_mean_mse": [],
            "baseline_mean_psnr": [],
            "baseline_mean_cosine": [],
            "attack": {},
            "attack_success": {},
        },
        "system": {
            "comm_bytes": [],
            "time_sec": [],
            "rounds": [],
        },
    }


def _compute_baseline_metrics(original: torch.Tensor, data_range: float = 1.0) -> Tuple[Dict[str, float], Dict[str, float]]:
    rng = np.random.default_rng(42)
    if original.dim() > 2:
        original = original.view(original.shape[0], -1)
    random_recon = torch.tensor(rng.random(original.shape), dtype=original.dtype)
    mean_img = original.mean(dim=0, keepdim=True)
    mean_recon = mean_img.repeat(original.shape[0], 1)

    def _metrics(target: torch.Tensor, recon: torch.Tensor) -> Dict[str, float]:
        mse = F.mse_loss(recon, target).item()
        psnr = 10 * np.log10((data_range**2) / (mse + 1e-10))
        cos = F.cosine_similarity(target.flatten().unsqueeze(0), recon.flatten().unsqueeze(0)).item()
        return {"mse": mse, "psnr": psnr, "cosine": cos}

    return _metrics(original, random_recon), _metrics(original, mean_recon)


def save_checkpoint(
    output_dir: Path,
    round_idx: int,
    lejepa_server: FederatedServer,
    mae_server: FederatedServer,
    results: Dict[str, Dict[str, List[float]]],
) -> Path:
    path = checkpoint_path(output_dir, round_idx)
    state = {
        "round": round_idx,
        "lejepa_state": lejepa_server.global_model.state_dict(),
        "mae_state": mae_server.global_model.state_dict(),
        "results": results,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, path)
    return path


def load_checkpoint(
    checkpoint_file: Path,
    lejepa_server: FederatedServer,
    mae_server: FederatedServer,
) -> Tuple[int, Dict[str, Dict[str, List[float]]]]:
    state = torch.load(checkpoint_file, map_location="cpu")
    lejepa_server.global_model.load_state_dict(state["lejepa_state"])
    mae_server.global_model.load_state_dict(state["mae_state"])
    results = state.get("results", _build_results_dict())
    if "torch_rng" in state:
        torch.set_rng_state(state["torch_rng"])
    if "numpy_rng" in state:
        np.random.set_state(state["numpy_rng"])
    if "python_rng" in state:
        random.setstate(state["python_rng"])
    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state.get("round", 0)), results


def run_federated_privacy_experiment(config: ExperimentConfig, output_dir: Path, logger) -> Dict[str, Dict[str, List[float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_samples = config.num_clients * config.samples_per_client
    mnist_transform = create_mnist_transform(config.image_shape)

    logger.info("=" * 60)
    logger.info("FEDERATED LEARNING PRIVACY COMPARISON")
    logger.info("LeJEPA vs. Masked Autoencoder")
    logger.info("Device: %s", device)
    if config.dp_config.enabled:
        logger.info(
            "DP Mode: enabled | clip_norm=%s | noise_multiplier=%s",
            config.dp_config.clip_norm,
            config.dp_config.noise_multiplier,
        )
    else:
        logger.info("DP Mode: disabled")
    logger.info("=" * 60)

    logger.info("[1] Loading MNIST and creating non-IID splits...")
    client_data, client_labels = load_mnist_non_iid(
        num_clients=config.num_clients,
        total_samples=total_samples,
        alpha=config.dirichlet_alpha,
        seed=config.seed,
        data_dir="data",
        image_shape=config.image_shape,
    )
    all_train_data = torch.cat(client_data)
    all_train_labels = torch.cat(client_labels)
    class_counts = plot_client_class_distribution(
        client_labels, num_classes=10, save_path=str(output_dir / "client_class_distribution.png")
    )
    for c in range(config.num_clients):
        logger.info(
            "Client %s: samples=%s, class_counts=%s",
            c,
            len(client_data[c]),
            class_counts[c].tolist(),
        )

    logger.info("[2] Initializing models...")
    lejepa_model = LeJEPAModel(
        config.input_dim,
        config.emb_dim,
        config.proj_dim,
        lamb=config.lamb,
        use_cnn=config.use_cnn,
        use_vit=config.use_vit,
        image_shape=config.image_shape,
    )
    if config.use_vit:
        mae_model = MAEViT(
            img_size=config.image_shape[1],
            mask_ratio=0.4,
            in_chans=config.image_shape[0],
            patch_size=16,
        )
    else:
        mae_model = MAEModel(
            config.input_dim,
            config.emb_dim,
            use_cnn=config.use_cnn,
            image_shape=config.image_shape,
        )
    lejepa_model.to(device)
    mae_model.to(device)
    logger.info("LeJEPA: proj_dim=%s, lambda=%s, cnn=%s, vit=%s", config.proj_dim, config.lamb, config.use_cnn, config.use_vit)
    logger.info("MAE: latent_dim=%s, cnn=%s, vit=%s", config.emb_dim, config.use_cnn, config.use_vit)

    lejepa_augmenter = ViewAugmenter(
        num_views=config.num_views,
        image_shape=config.image_shape,
        device=device,
        normalize_mean=config.normalize_mean,
        normalize_std=config.normalize_std,
        **config.augmenter_kwargs,
    )
    mae_augmenter = None
    mae_aug_kwargs = {}
    if config.align_augmentations:
        mae_aug_kwargs.update(config.augmenter_kwargs)
    mae_aug_kwargs.update(config.mae_augmenter_kwargs)
    mae_aug_enabled = config.align_augmentations or mae_aug_kwargs.get("enabled", False)
    mae_aug_kwargs.pop("enabled", None)
    if mae_aug_enabled:
        mae_augmenter = ViewAugmenter(
            num_views=1,
            image_shape=config.image_shape,
            device=device,
            normalize_mean=config.normalize_mean,
            normalize_std=config.normalize_std,
            **mae_aug_kwargs,
        )
    else:
        mae_augmenter = IdentityAugmenter(config.normalize_mean, config.normalize_std)

    attack_augmenter = ViewAugmenter(
        num_views=config.num_views,
        image_shape=config.image_shape,
        device=device,
        normalize_mean=config.normalize_mean,
        normalize_std=config.normalize_std,
        deterministic=config.attack_deterministic_augment,
        base_seed=config.attack_seed,
        **config.augmenter_kwargs,
    )
    attack_mae_augmenter = ViewAugmenter(
        num_views=1,
        image_shape=config.image_shape,
        device=device,
        normalize_mean=config.normalize_mean,
        normalize_std=config.normalize_std,
        deterministic=config.attack_deterministic_augment,
        base_seed=config.attack_seed,
        **mae_aug_kwargs,
    )

    lejepa_clients = [
        FederatedClient(
            i,
            client_data[i],
            client_labels[i],
            "lejepa",
            config.num_views,
            device=device,
            dp_config=config.dp_config,
            augmenter=lejepa_augmenter,
        )
        for i in range(config.num_clients)
    ]
    mae_clients = [
        FederatedClient(
            i,
            client_data[i],
            client_labels[i],
            "mae",
            1,
            device=device,
            dp_config=config.dp_config,
            augmenter=mae_augmenter,
        )
        for i in range(config.num_clients)
    ]

    lejepa_server = FederatedServer(lejepa_model)
    mae_server = FederatedServer(mae_model)

    if config.attack_on == "gradients":
        lejepa_attacker = GradientInversionAttack(lejepa_model, config.input_dim, config.num_views)
        mae_attacker = GradientInversionAttack(mae_model, config.input_dim, 1)
    else:
        lejepa_attacker = UpdateInversionAttack(
            lejepa_model,
            config.input_dim,
            config.num_views,
            view_augmenter=None,
            normalize_mean=config.normalize_mean,
            normalize_std=config.normalize_std,
        )
        mae_attacker = UpdateInversionAttack(
            mae_model,
            config.input_dim,
            1,
            view_augmenter=None,
            normalize_mean=config.normalize_mean,
            normalize_std=config.normalize_std,
        )
    metrics_helper = GradientInversionAttack(lejepa_model, config.input_dim, config.num_views)

    results = _build_results_dict()

    loss_log_path = output_dir / "loss_components_log.csv"
    initialize_loss_log(loss_log_path)
    attack_class_log_path = output_dir / "attack_per_class_metrics.csv"
    initialize_attack_class_log(attack_class_log_path)

    start_round = 0
    if config.resume_from:
        checkpoint_file = Path(config.resume_from)
        if checkpoint_file.exists():
            start_round, results = load_checkpoint(checkpoint_file, lejepa_server, mae_server)
            logger.info("Resumed from checkpoint %s (round %s)", checkpoint_file, start_round)

    last_reconstructions: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    logger.info("[3] Running federated training...")
    for round_idx in range(start_round, config.num_rounds):
        logger.info("Round %s/%s", round_idx + 1, config.num_rounds)
        round_start = time.perf_counter()

        lejepa_updates = [
            client.local_train(
                lejepa_server.global_model,
                epochs=config.local_epochs,
                lr=config.learning_rate,
                batch_size=config.batch_size,
                max_batches=config.max_batches_per_epoch,
                optimizer_name=config.optimizer,
            )
            for client in lejepa_clients
        ]
        lejepa_server.aggregate(lejepa_updates)
        lejepa_avg_loss = float(np.mean([update["loss"] for update in lejepa_updates]))
        results["lejepa"]["loss"].append(lejepa_avg_loss)
        results["lejepa"]["loss_rounds"].append(round_idx)

        lejepa_avg_inv = float(np.mean([u["inv_loss"] for u in lejepa_updates if u["inv_loss"] is not None]))
        lejepa_avg_sigreg = float(
            np.mean([u["sigreg_loss"] for u in lejepa_updates if u["sigreg_loss"] is not None])
        )
        results["lejepa"]["inv"].append(lejepa_avg_inv)
        results["lejepa"]["sigreg"].append(lejepa_avg_sigreg)

        for client_id, update in enumerate(lejepa_updates):
            append_loss_log(
                loss_log_path,
                round_idx,
                "lejepa",
                "client",
                client_id,
                {
                    "total": update["loss"],
                    "inv": update.get("inv_loss"),
                    "sigreg": update.get("sigreg_loss"),
                },
            )

        mae_updates = [
            client.local_train(
                mae_server.global_model,
                epochs=config.local_epochs,
                lr=config.learning_rate,
                batch_size=config.batch_size,
                max_batches=config.max_batches_per_epoch,
                optimizer_name=config.optimizer,
            )
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

        round_time = time.perf_counter() - round_start
        num_params = sum(p.numel() for p in lejepa_server.global_model.parameters())
        bytes_per_model = num_params * 4
        comm_bytes = config.num_clients * bytes_per_model * 2
        results["system"]["comm_bytes"].append(float(comm_bytes))
        results["system"]["time_sec"].append(float(round_time))
        results["system"]["rounds"].append(round_idx)

        global_batch = sample_tensor_batch(all_train_data, max_samples=256, seed=round_idx + 1)
        lejepa_global_losses = compute_loss_components(
            lejepa_server.global_model,
            global_batch,
            model_type="lejepa",
            augmenter=lejepa_augmenter,
        )
        results["lejepa"]["global_loss"].append(lejepa_global_losses["total"])
        results["lejepa"]["global_inv"].append(lejepa_global_losses["inv"])
        results["lejepa"]["global_sigreg"].append(lejepa_global_losses["sigreg"])
        results["lejepa"]["global_rounds"].append(round_idx)
        append_loss_log(
            loss_log_path,
            round_idx,
            "lejepa",
            "global",
            -1,
            lejepa_global_losses,
        )

        mae_global_losses = compute_loss_components(
            mae_server.global_model,
            global_batch,
            model_type="mae",
            augmenter=mae_augmenter,
        )
        results["mae"]["global_loss"].append(mae_global_losses["total"])
        results["mae"]["global_rounds"].append(round_idx)
        append_loss_log(
            loss_log_path,
            round_idx,
            "mae",
            "global",
            -1,
            mae_global_losses,
        )

        if round_idx % config.eval_every == 0 or round_idx == config.num_rounds - 1:
            max_clients = min(config.attack_eval_clients, config.num_clients)
            client_indices = list(range(max_clients))
            strategies = [s.lower().strip() for s in config.attack_loss_strategies]

            for strategy in strategies:
                results["lejepa"]["attack"].setdefault(
                    strategy, {"mse": [], "psnr": [], "cosine": [], "rounds": []}
                )
                results["mae"]["attack"].setdefault(
                    strategy, {"mse": [], "psnr": [], "cosine": [], "rounds": []}
                )
                results["lejepa"]["attack_success"].setdefault(
                    strategy, {"rate": [], "rounds": []}
                )
                results["mae"]["attack_success"].setdefault(
                    strategy, {"rate": [], "rounds": []}
                )

            for model_key, updates, attacker in (
                ("lejepa", lejepa_updates, lejepa_attacker),
                ("mae", mae_updates, mae_attacker),
            ):
                attacker_augmenter = attack_augmenter if model_key == "lejepa" else attack_mae_augmenter
                for strategy in strategies:
                    metric_vals = {"mse": [], "psnr": [], "cosine": []}
                    success_vals = []
                    for client_idx in client_indices:
                        update = updates[client_idx]
                        if config.attack_use_raw_data:
                            attack_input = update["raw_data"]
                            denorm_fn = None
                        else:
                            attack_input = update["data"]
                            denorm_fn = lambda x: denormalize_mnist(
                                x, config.normalize_mean, config.normalize_std
                            )

                        attack_batches = [(attack_input, update)]
                        extra_batches = max(0, config.attack_eval_batches - 1)
                        for extra_idx in range(extra_batches):
                            extra_batch = sample_tensor_batch(
                                client_data[client_idx],
                                max_samples=config.batch_size,
                                seed=round_idx + client_idx + extra_idx + 1,
                            )
                            if not config.attack_use_raw_data:
                                extra_batch = normalize_mnist(
                                    extra_batch, config.normalize_mean, config.normalize_std
                                )
                            attack_batches.append((extra_batch, None))

                        for attack_input, update_obj in attack_batches:
                            if config.attack_on == "gradients":
                                true_signal = (
                                    update_obj["gradients"]
                                    if update_obj is not None
                                    else compute_gradients_for_data(
                                        lejepa_server.global_model if model_key == "lejepa" else mae_server.global_model,
                                        attack_input,
                                        model_type=model_key,
                                        num_views=config.num_views if model_key == "lejepa" else 1,
                                        augmenter=attacker_augmenter,
                                    )
                                )
                            else:
                                true_signal = (
                                    update_obj["update_vector"]
                                    if update_obj is not None
                                    else attacker._compute_update_vector(
                                        attack_input.to(device), lr=config.learning_rate
                                    )
                                )

                            if config.attack_on == "gradients":
                                recon, _ = attacker.attack(
                                    attack_input,
                                    true_signal,
                                    lr=0.05,
                                    iterations=600,
                                    loss_strategy=strategy,
                                )
                            else:
                                recon, _ = attacker.attack(
                                    attack_input,
                                    true_signal,
                                    lr=0.05,
                                    iterations=600,
                                    loss_strategy=strategy,
                                    update_lr=config.learning_rate,
                                )

                            metrics = metrics_helper.compute_metrics(
                                attack_input,
                                recon,
                                denormalize_fn=denorm_fn,
                            )
                            metric_vals["mse"].append(metrics["mse"])
                            metric_vals["psnr"].append(metrics["psnr"])
                            metric_vals["cosine"].append(metrics["cosine_sim"])
                            success_vals.append(
                                1.0 if metrics["mse"] <= config.attack_success_mse_threshold else 0.0
                            )

                            if (
                                round_idx == config.num_rounds - 1
                                and strategy == strategies[0]
                                and update_obj is not None
                            ):
                                last_reconstructions[f"{model_key}_attack"] = (
                                    attack_input.detach(), recon.detach()
                                )

                    avg_mse = float(np.mean(metric_vals["mse"]))
                    avg_psnr = float(np.mean(metric_vals["psnr"]))
                    avg_cos = float(np.mean(metric_vals["cosine"]))
                    success_rate = float(np.mean(success_vals))

                    results[model_key]["attack"][strategy]["mse"].append(avg_mse)
                    results[model_key]["attack"][strategy]["psnr"].append(avg_psnr)
                    results[model_key]["attack"][strategy]["cosine"].append(avg_cos)
                    results[model_key]["attack"][strategy]["rounds"].append(round_idx)
                    results[model_key]["attack_success"][strategy]["rate"].append(success_rate)
                    results[model_key]["attack_success"][strategy]["rounds"].append(round_idx)

                    if strategy == strategies[0]:
                        results[model_key]["mse"].append(avg_mse)
                        results[model_key]["psnr"].append(avg_psnr)
                        results[model_key]["cosine"].append(avg_cos)
                        results[model_key]["rounds"].append(round_idx)

            baseline_input = (
                lejepa_updates[client_indices[0]]["raw_data"]
                if config.attack_use_raw_data
                else denormalize_mnist(
                    lejepa_updates[client_indices[0]]["data"],
                    config.normalize_mean,
                    config.normalize_std,
                )
            )
            baseline_random, baseline_mean = _compute_baseline_metrics(baseline_input)
            results["lejepa"]["baseline_random_mse"].append(baseline_random["mse"])
            results["lejepa"]["baseline_random_psnr"].append(baseline_random["psnr"])
            results["lejepa"]["baseline_random_cosine"].append(baseline_random["cosine"])
            results["lejepa"]["baseline_mean_mse"].append(baseline_mean["mse"])
            results["lejepa"]["baseline_mean_psnr"].append(baseline_mean["psnr"])
            results["lejepa"]["baseline_mean_cosine"].append(baseline_mean["cosine"])

            baseline_random_mae, baseline_mean_mae = _compute_baseline_metrics(baseline_input)
            results["mae"]["baseline_random_mse"].append(baseline_random_mae["mse"])
            results["mae"]["baseline_random_psnr"].append(baseline_random_mae["psnr"])
            results["mae"]["baseline_random_cosine"].append(baseline_random_mae["cosine"])
            results["mae"]["baseline_mean_mse"].append(baseline_mean_mae["mse"])
            results["mae"]["baseline_mean_psnr"].append(baseline_mean_mae["psnr"])
            results["mae"]["baseline_mean_cosine"].append(baseline_mean_mae["cosine"])

            logger.info(
                "[Eval] Attack(%s) JEPA MSE=%.4f PSNR=%.2f Cos=%.4f | MAE MSE=%.4f PSNR=%.2f Cos=%.4f",
                strategies[0],
                results["lejepa"]["mse"][-1],
                results["lejepa"]["psnr"][-1],
                results["lejepa"]["cosine"][-1],
                results["mae"]["mse"][-1],
                results["mae"]["psnr"][-1],
                results["mae"]["cosine"][-1],
            )

            primary_strategy = strategies[0]
            class_samples = sample_class_images(
                all_train_data,
                all_train_labels,
                config.plot_classes,
                samples_per_class=1,
                seed=config.seed + round_idx,
            )
            for cls, samples in class_samples.items():
                plot_samples = samples.clone()
                for model_key, model, attacker, augmenter in (
                    ("lejepa", lejepa_server.global_model, lejepa_attacker, lejepa_augmenter),
                    ("mae", mae_server.global_model, mae_attacker, mae_augmenter),
                ):
                    if config.attack_on == "gradients":
                        signal = compute_gradients_for_data(
                            model,
                            plot_samples,
                            model_type=model_key,
                            num_views=config.num_views if model_key == "lejepa" else 1,
                            augmenter=augmenter,
                        )
                    else:
                        signal = attacker._compute_update_vector(
                            plot_samples.to(device), lr=config.learning_rate
                        )

                    if config.attack_on == "gradients":
                        recon, _ = attacker.attack(
                            plot_samples,
                            signal,
                            iterations=200,
                            loss_strategy=primary_strategy,
                        )
                    else:
                        recon, _ = attacker.attack(
                            plot_samples,
                            signal,
                            iterations=200,
                            loss_strategy=primary_strategy,
                            update_lr=config.learning_rate,
                        )

                    metrics = metrics_helper.compute_metrics(
                        plot_samples,
                        recon,
                        denormalize_fn=None,
                    )
                    success = 1.0 if metrics["mse"] <= config.attack_success_mse_threshold else 0.0
                    append_attack_class_log(
                        attack_class_log_path,
                        round_idx,
                        model_key,
                        primary_strategy,
                        int(cls),
                        metrics,
                        success,
                    )

        if config.plot_rounds and round_idx in config.plot_rounds:
            logger.info("[Plotting] Reconstruction steps at round %s", round_idx + 1)
            class_samples = sample_class_images(
                all_train_data,
                all_train_labels,
                config.plot_classes,
                samples_per_class=4,
                seed=config.seed + round_idx,
            )

            histories = {"lejepa": {}, "mae": {}}
            originals = {"lejepa": {}, "mae": {}}

            for cls, samples in class_samples.items():
                plot_samples = samples.clone()
                if config.attack_on == "gradients":
                    lejepa_signal = compute_gradients_for_data(
                        lejepa_server.global_model,
                        samples,
                        model_type="lejepa",
                        num_views=config.num_views,
                        augmenter=lejepa_augmenter,
                    )
                    mae_signal = compute_gradients_for_data(
                        mae_server.global_model,
                        samples,
                        model_type="mae",
                        num_views=1,
                        augmenter=mae_augmenter,
                    )
                else:
                    lejepa_signal = lejepa_attacker._compute_update_vector(
                        samples.to(device), lr=config.learning_rate
                    )
                    mae_signal = mae_attacker._compute_update_vector(
                        samples.to(device), lr=config.learning_rate
                    )

                if config.attack_on == "gradients":
                    lejepa_recon, lejepa_history = lejepa_attacker.attack(
                        samples,
                        lejepa_signal,
                        iterations=max(config.plot_steps),
                        return_history=True,
                        record_steps=config.plot_steps,
                    )
                    mae_recon, mae_history = mae_attacker.attack(
                        samples,
                        mae_signal,
                        iterations=max(config.plot_steps),
                        return_history=True,
                        record_steps=config.plot_steps,
                    )
                else:
                    lejepa_recon, lejepa_history = lejepa_attacker.attack(
                        samples,
                        lejepa_signal,
                        iterations=max(config.plot_steps),
                        return_history=True,
                        record_steps=config.plot_steps,
                        update_lr=config.learning_rate,
                    )
                    mae_recon, mae_history = mae_attacker.attack(
                        samples,
                        mae_signal,
                        iterations=max(config.plot_steps),
                        return_history=True,
                        record_steps=config.plot_steps,
                        update_lr=config.learning_rate,
                    )

                histories["lejepa"][cls] = lejepa_history
                histories["mae"][cls] = mae_history
                originals["lejepa"][cls] = plot_samples
                originals["mae"][cls] = plot_samples.clone()

            plot_reconstruction_steps_by_class(
                originals_by_class=originals["lejepa"],
                histories_by_class=histories["lejepa"],
                steps=config.plot_steps,
                save_path=str(output_dir / f"lejepa_recon_steps_round{round_idx + 1}.png"),
                title=f"LeJEPA Recon Steps (Round {round_idx + 1})",
                image_shape=config.image_shape,
                normalize_mean=config.normalize_mean,
                normalize_std=config.normalize_std,
                denormalize=not config.attack_use_raw_data,
            )
            plot_reconstruction_steps_by_class(
                originals_by_class=originals["mae"],
                histories_by_class=histories["mae"],
                steps=config.plot_steps,
                save_path=str(output_dir / f"mae_recon_steps_round{round_idx + 1}.png"),
                title=f"MAE Recon Steps (Round {round_idx + 1})",
                image_shape=config.image_shape,
                normalize_mean=config.normalize_mean,
                normalize_std=config.normalize_std,
                denormalize=not config.attack_use_raw_data,
            )

            logger.info("[Plotting] t-SNE embeddings for validation samples")
            val_dataset = datasets.MNIST(root="data", train=False, download=True, transform=mnist_transform)
            val_data, val_labels = sample_mnist_dataset(val_dataset, max_samples=600)
            val_data_norm = normalize_mnist(val_data, config.normalize_mean, config.normalize_std)
            plot_tsne_for_validation(
                lejepa_server.global_model,
                val_data_norm,
                val_labels,
                model_type="lejepa",
                save_path=str(output_dir / f"lejepa_tsne_round{round_idx + 1}_val.png"),
                title=f"LeJEPA Validation t-SNE (Round {round_idx + 1})",
                image_shape=config.image_shape,
            )
            plot_tsne_for_validation(
                mae_server.global_model,
                val_data_norm,
                val_labels,
                model_type="mae",
                save_path=str(output_dir / f"mae_tsne_round{round_idx + 1}_val.png"),
                title=f"MAE Validation t-SNE (Round {round_idx + 1})",
                image_shape=config.image_shape,
            )

        if config.plot_rounds and round_idx in config.plot_rounds:
            logger.info("[Probing] Linear probe at round %s", round_idx + 1)
            train_data, train_labels = sample_tensor_dataset(
                all_train_data, all_train_labels, max_samples=2000
            )

            test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=mnist_transform)
            test_data, test_labels = sample_mnist_dataset(test_dataset, max_samples=1000)

            train_data_norm = normalize_mnist(train_data, config.normalize_mean, config.normalize_std)
            test_data_norm = normalize_mnist(test_data, config.normalize_mean, config.normalize_std)

            lejepa_probe_acc = train_linear_probe(
                lejepa_server.global_model,
                train_data_norm,
                train_labels,
                test_data_norm,
                test_labels,
                model_type="lejepa",
                image_shape=config.image_shape,
            )
            mae_probe_acc = train_linear_probe(
                mae_server.global_model,
                train_data_norm,
                train_labels,
                test_data_norm,
                test_labels,
                model_type="mae",
                image_shape=config.image_shape,
            )
            results["lejepa"]["probe_acc"].append(lejepa_probe_acc)
            results["lejepa"]["probe_rounds"].append(round_idx)
            results["mae"]["probe_acc"].append(mae_probe_acc)
            results["mae"]["probe_rounds"].append(round_idx)
            logger.info("Probe JEPA: %.2f%%", lejepa_probe_acc * 100)
            logger.info("Probe MAE: %.2f%%", mae_probe_acc * 100)

        if config.checkpoint_every > 0 and (round_idx + 1) % config.checkpoint_every == 0:
            ckpt = save_checkpoint(output_dir, round_idx, lejepa_server, mae_server, results)
            logger.info("Saved checkpoint: %s", ckpt)

    if last_reconstructions:
        logger.info("[4] Saving reconstructed image grids...")
        lejepa_orig, lejepa_recon = last_reconstructions["lejepa_attack"]
        mae_orig, mae_recon = last_reconstructions["mae_attack"]
        plot_reconstructions(
            lejepa_orig,
            lejepa_recon,
            save_path=str(output_dir / "lejepa_reconstructions.png"),
            title="LeJEPA Update Inversion",
            image_shape=config.image_shape,
            normalize_mean=config.normalize_mean,
            normalize_std=config.normalize_std,
            denormalize=not config.attack_use_raw_data,
        )
        plot_reconstructions(
            mae_orig,
            mae_recon,
            save_path=str(output_dir / "mae_reconstructions.png"),
            title="MAE Update Inversion",
            image_shape=config.image_shape,
            normalize_mean=config.normalize_mean,
            normalize_std=config.normalize_std,
            denormalize=not config.attack_use_raw_data,
        )

    logger.info("[5] Plotting learning curves...")
    plot_metric_curve(
        results["lejepa"]["loss_rounds"],
        results["lejepa"]["loss"],
        results["mae"]["loss"],
        label_a="LeJEPA",
        label_b="MAE",
        ylabel="Local training loss",
        title="Federated Training Loss",
        save_path=str(output_dir / "training_loss_curve.png"),
    )

    plot_scalar_curve(
        results["system"]["rounds"],
        results["system"]["comm_bytes"],
        ylabel="Communication (bytes)",
        title="Per-Round Communication Cost",
        save_path=str(output_dir / "communication_cost_curve.png"),
    )
    plot_scalar_curve(
        results["system"]["rounds"],
        results["system"]["time_sec"],
        ylabel="Seconds",
        title="Per-Round Training Time",
        save_path=str(output_dir / "training_time_curve.png"),
    )

    lejepa_probe_map = dict(zip(results["lejepa"]["probe_rounds"], results["lejepa"]["probe_acc"]))
    mae_probe_map = dict(zip(results["mae"]["probe_rounds"], results["mae"]["probe_acc"]))

    lejepa_trade_priv = []
    lejepa_trade_util = []
    for round_id, mse in zip(results["lejepa"]["rounds"], results["lejepa"]["mse"]):
        if round_id in lejepa_probe_map:
            lejepa_trade_priv.append(mse)
            lejepa_trade_util.append(lejepa_probe_map[round_id])

    mae_trade_priv = []
    mae_trade_util = []
    for round_id, mse in zip(results["mae"]["rounds"], results["mae"]["mse"]):
        if round_id in mae_probe_map:
            mae_trade_priv.append(mse)
            mae_trade_util.append(mae_probe_map[round_id])

    plot_tradeoff_curve(
        lejepa_trade_priv,
        lejepa_trade_util,
        label="LeJEPA",
        xlabel="Privacy (MSE)",
        ylabel="Probe Accuracy",
        title="Utility-Privacy Tradeoff",
        save_path=str(output_dir / "utility_privacy_tradeoff_lejepa.png"),
    )
    plot_tradeoff_curve(
        mae_trade_priv,
        mae_trade_util,
        label="MAE",
        xlabel="Privacy (MSE)",
        ylabel="Probe Accuracy",
        title="Utility-Privacy Tradeoff",
        save_path=str(output_dir / "utility_privacy_tradeoff_mae.png"),
    )

    logger.info("[6] Statistical summaries...")
    for metric in ("mse", "psnr", "cosine"):
        lejepa_vals = results["lejepa"][metric]
        mae_vals = results["mae"][metric]
        ci_low, ci_high = bootstrap_ci(lejepa_vals)
        mae_ci_low, mae_ci_high = bootstrap_ci(mae_vals)
        p_value = permutation_test(lejepa_vals, mae_vals)

        logger.info(
            "%s | LeJEPA mean=%.4f CI=[%.4f, %.4f] | MAE mean=%.4f CI=[%.4f, %.4f] | p=%.4f",
            metric.upper(),
            summarize_metric(lejepa_vals)["mean"],
            ci_low,
            ci_high,
            summarize_metric(mae_vals)["mean"],
            mae_ci_low,
            mae_ci_high,
            p_value,
        )

        baseline_random_key = f"baseline_random_{metric}"
        baseline_mean_key = f"baseline_mean_{metric}"
        logger.info(
            "%s baselines | random=%.4f | mean=%.4f",
            metric.upper(),
            summarize_metric(results["lejepa"][baseline_random_key])["mean"],
            summarize_metric(results["lejepa"][baseline_mean_key])["mean"],
        )

    return results
