"""Utilities for reproducibility, logging, and filesystem management."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        if torch.cuda.is_available():
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except (AttributeError, RuntimeError):
            pass


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(output_dir: str | Path, log_name: str = "run.log") -> logging.Logger:
    output_dir = ensure_dir(output_dir)
    log_path = output_dir / log_name

    logger = logging.getLogger("mnist_experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def disable_cuda_sdp_kernels() -> None:
    """Disable CUDA scaled dot-product attention kernels without backward support."""
    if not torch.cuda.is_available():
        return
    backend = torch.backends.cuda
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
                setattr(backend, attr, False)


def checkpoint_path(output_dir: Path, round_idx: int) -> Path:
    return output_dir / f"checkpoint_round_{round_idx:06d}.pt"
