"""Entry point for MNIST federated privacy experiment."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import ExperimentConfig, load_config
from .training import run_federated_privacy_experiment
from .utils import disable_cuda_sdp_kernels, ensure_dir, resolve_device, set_seed, setup_logging


def _build_output_dir(base_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(base_dir) / f"mnist_{timestamp}")


def run() -> None:
    parser = argparse.ArgumentParser(description="MNIST federated privacy experiment")
    parser.add_argument("--config", type=str, default="configs/mnist.json", help="Path to JSON config file")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume")
    args = parser.parse_args()

    config: ExperimentConfig = load_config(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.resume_from:
        config.resume_from = args.resume_from

    if not config.plot_rounds:
        config.plot_rounds = np.linspace(0, config.num_rounds - 1, 20, dtype=int).tolist()

    output_dir = _build_output_dir(config.output_dir)
    logger = setup_logging(output_dir)

    device = resolve_device(config.device)
    config.device = str(device)

    disable_cuda_sdp_kernels()

    set_seed(config.seed, deterministic=config.deterministic)

    logger.info("Config: %s", config.to_dict())

    run_federated_privacy_experiment(config, output_dir, logger)
