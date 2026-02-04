"""Dataset helpers for MNIST federated experiments."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torchvision import datasets, transforms


def create_mnist_transform(image_shape: Tuple[int, int, int]) -> transforms.Compose:
    """Create MNIST transform with resizing to target image shape."""
    _, height, width = image_shape
    return transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ]
    )


def normalize_mnist(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (x - mean) / std


def denormalize_mnist(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (x * std) + mean


def load_mnist_non_iid(
    num_clients: int,
    total_samples: int,
    alpha: float = 0.5,
    seed: int = 42,
    data_dir: str = "data",
    image_shape: Tuple[int, int, int] = (1, 28, 28),
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
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
        _, label = dataset[int(idx)]
        class_indices[int(label)].append(int(idx))

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
            client_indices[client_id].extend(idxs[cursor : cursor + count])
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
            img, label = dataset[int(idx)]
            images.append(img.view(-1))
            labels.append(int(label))
        client_data.append(torch.stack(images))
        client_labels.append(torch.tensor(labels, dtype=torch.long))

    return client_data, client_labels


def sample_tensor_dataset(
    data: torch.Tensor,
    labels: torch.Tensor,
    max_samples: int,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly sample up to max_samples from tensors."""
    if len(data) <= max_samples:
        return data, labels
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data))[:max_samples]
    return data[indices], labels[indices]


def sample_mnist_dataset(
    dataset: datasets.MNIST,
    max_samples: int,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def sample_tensor_batch(data: torch.Tensor, max_samples: int, seed: int = 42) -> torch.Tensor:
    """Sample up to max_samples from a tensor."""
    if len(data) <= max_samples:
        return data
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data))[:max_samples]
    return data[indices]


def sample_class_images(
    data: torch.Tensor,
    labels: torch.Tensor,
    classes: List[int],
    samples_per_class: int = 4,
) -> Dict[int, torch.Tensor]:
    """Return a dict mapping class -> samples (N, 784)."""
    class_samples = {}
    for cls in classes:
        indices = (labels == cls).nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            continue
        selected = indices[:samples_per_class]
        class_samples[int(cls)] = data[selected]
    return class_samples
