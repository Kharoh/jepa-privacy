"""Plotting utilities for MNIST experiment."""

from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from sklearn.manifold import TSNE

from .data import denormalize_mnist


def plot_client_class_distribution(
    client_labels: List[torch.Tensor],
    num_classes: int = 10,
    save_path: str = "client_class_distribution.png",
) -> np.ndarray:
    """Plot and save the per-client class distribution heatmap."""
    counts = np.zeros((len(client_labels), num_classes), dtype=int)
    for client_id, labels in enumerate(client_labels):
        unique, freqs = np.unique(labels.numpy(), return_counts=True)
        counts[client_id, unique] = freqs

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(counts, aspect="auto", cmap="viridis")
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


def _resolve_image_hw(
    image_shape: Tuple[int, int, int] | None = None,
    flat_dim: int | None = None,
) -> Tuple[int, int]:
    """Resolve image height/width from an image_shape or flattened dimension."""
    if image_shape is not None:
        return image_shape[1], image_shape[2]
    if flat_dim is None:
        raise ValueError("flat_dim is required when image_shape is not provided")
    side = int(np.sqrt(flat_dim))
    if side * side != flat_dim:
        raise ValueError(f"Cannot infer square image size from flat_dim={flat_dim}")
    return side, side


def plot_reconstructions(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    save_path: str,
    title: str,
    num_images: int = 8,
    image_shape: Tuple[int, int, int] | None = None,
    normalize_mean: float = 0.1307,
    normalize_std: float = 0.3081,
    denormalize: bool = True,
) -> None:
    """Plot original vs reconstructed images in a 2xN grid."""
    if original.dim() == 3:
        original = original[:, 0, :]
    if reconstructed.dim() == 3:
        reconstructed = reconstructed[:, 0, :]

    if denormalize:
        original = denormalize_mnist(original, normalize_mean, normalize_std)
        reconstructed = denormalize_mnist(reconstructed, normalize_mean, normalize_std)

    height, width = _resolve_image_hw(image_shape, flat_dim=original.shape[-1])
    original = original[:num_images].reshape(-1, height, width).cpu()
    reconstructed = reconstructed[:num_images].reshape(-1, height, width).cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.4, 3.2))
    for i in range(num_images):
        axes[0, i].imshow(original[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed[i], cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_reconstruction_steps_by_class(
    originals_by_class: Dict[int, torch.Tensor],
    histories_by_class: Dict[int, Dict[int, torch.Tensor]],
    steps: List[int],
    save_path: str,
    title: str,
    image_shape: Tuple[int, int, int] | None = None,
    normalize_mean: float = 0.1307,
    normalize_std: float = 0.3081,
    denormalize: bool = True,
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
        if denormalize:
            original = denormalize_mnist(original, normalize_mean, normalize_std)
        height, width = _resolve_image_hw(image_shape, flat_dim=original.shape[-1])
        img = original[0].reshape(height, width).cpu()
        axes[0, col].imshow(img, cmap="gray")
        axes[0, col].set_title(f"Class {cls}")
        axes[0, col].axis("off")

        history = histories_by_class.get(cls, {})
        for row, step in enumerate(steps, start=1):
            recon = history.get(step)
            if recon is None:
                axes[row, col].axis("off")
                continue
            if recon.dim() == 3:
                recon = recon[:, 0, :]
            if denormalize:
                recon = denormalize_mnist(recon, normalize_mean, normalize_std)
            img = recon[0].reshape(height, width).cpu()
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(f"Step {step}", fontsize=8)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_tsne_latents(latents: torch.Tensor, labels: torch.Tensor, save_path: str, title: str) -> None:
    """Plot t-SNE embedding with class labels."""
    if latents.shape[0] < 2:
        return

    perplexity = min(30, max(2, latents.shape[0] - 1))
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=42,
    )
    embedding = tsne.fit_transform(latents.numpy())

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.get_cmap("tab10")
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels.numpy(),
        cmap=cmap,
        alpha=0.8,
        s=18,
    )
    unique_labels = torch.unique(labels).tolist()
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=cmap(int(label) % cmap.N),
            label=str(int(label)),
            markersize=6,
        )
        for label in unique_labels
    ]
    legend = ax.legend(
        handles=handles,
        title="Class",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
    )
    ax.add_artist(legend)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_metric_curve(
    x: List[int],
    y_a: List[float],
    y_b: List[float],
    label_a: str,
    label_b: str,
    ylabel: str,
    title: str,
    save_path: str,
) -> None:
    """Plot a two-line curve for JEPA/MAE metrics."""
    if len(x) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y_a, marker="o", label=label_a)
    ax.plot(x, y_b, marker="o", label=label_b)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_tradeoff_curve(
    privacy_vals: List[float],
    utility_vals: List[float],
    label: str,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str,
) -> None:
    if len(privacy_vals) == 0 or len(utility_vals) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(privacy_vals, utility_vals, alpha=0.8, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_scalar_curve(
    x: List[int],
    y: List[float],
    ylabel: str,
    title: str,
    save_path: str,
) -> None:
    if len(x) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
