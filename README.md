# LeJEPA vs MAE Federated Privacy (ImageNette + MNIST)

This experiment compares LeJEPA and MAE on the MNIST dataset using ViT backbones in a federated learning setup. It runs gradient inversion attacks during training and reporting privacy metrics plus online probe accuracy for both models.

The MNIST pipeline is the reference implementation for privacy analysis and includes
update-based inversion (FedAvg-style), aligned augmentations, per-round utility metrics, and
communication/time tracking.

## What’s inside
- MNIST dataset loading
- Federated learning with non-IID client splits (Dirichlet)
- LeJEPA with SIGReg + invariance loss on a ViT-S/8 backbone
- MAE with a ViT-S/8 encoder and lightweight decoder for patch reconstruction
- Gradient inversion attacks and privacy metrics (MSE/PSNR/Cosine)
- Online linear probe accuracy for both methods

## How to run
1. Install dependencies from `requirements.txt`.
2. Run the script you want:
	- MNIST (with JSON config): `scripts/run_mnist.py --config configs/mnist.json`

## MNIST experiment protocol (summary)
- **Federated mode:** FedAvg with model update sharing by default; set `federated_strategy` to `gradients` to use averaged gradients (FedSGD-style).
- **Attack signal:** update vectors (one-step SGD by default via `max_batches_per_epoch=1`).
- **Augmentation alignment:** MAE uses the same augmentation family as LeJEPA (single view).
- **Reconstruction target:** raw MNIST pixels in $[0,1]$ (optionally normalized if configured).
- **Reproducibility:** deterministic attack augmentations can be enabled via config.

## Output artifacts
MNIST results are written under `results/mnist_<timestamp>/` (configurable). Example files include:
- `client_class_distribution.png`: class distribution per client
- `lejepa_reconstructions.png`: JEPA update inversion reconstructions
- `mae_reconstructions.png`: MAE update inversion reconstructions
- `training_loss_curve.png`: LeJEPA/MAE training loss over rounds
- `linear_probe_curve.png`: LeJEPA/MAE probe accuracy over rounds
- `utility_privacy_tradeoff_lejepa.png`: privacy vs probe accuracy (LeJEPA)
- `utility_privacy_tradeoff_mae.png`: privacy vs probe accuracy (MAE)
- `communication_cost_curve.png`: per-round communication cost
- `training_time_curve.png`: per-round training time
- `attack_per_class_metrics.csv`: per-class attack metrics by round
- `loss_components_log.csv`: per-round loss components for each client

## Project layout
- `src/lejepa_privacy/experiments/mnist_experiment/`: MNIST pipeline modules (data/models/training/privacy/utils).
- `src/lejepa_privacy/experiments/mnist.py`: MNIST entrypoint.
- `scripts/`: convenience entrypoints.

## Notes
- Adjust settings in `configs/mnist.json` (or pass a custom config path).
- The pipeline uses CUDA automatically when available and falls back to CPU otherwise.
- For reproducibility, the seed and deterministic settings are configurable in the config file.
- For attack reproducibility, set `attack_deterministic_augment` and `attack_seed`.
