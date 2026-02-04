# LeJEPA vs MAE Federated Privacy (ImageNette)

This experiment compares LeJEPA and MAE on the ImageNette (inet10) dataset using ViT backbones
in a federated learning setup. It mirrors the original MNIST privacy pipeline by running
gradient inversion attacks during training and reporting privacy metrics plus online probe
accuracy for both models.

## What’s inside
- ImageNette dataset loading via Hugging Face Datasets
- Federated learning with non-IID client splits (Dirichlet)
- LeJEPA with SIGReg + invariance loss on a ViT-S/8 backbone
- MAE with a ViT-S/8 encoder and lightweight decoder for patch reconstruction
- Gradient inversion attacks and privacy metrics (MSE/PSNR/Cosine)
- Online linear probe accuracy for both methods

## How to run
1. Install dependencies from `requirements.txt`.
2. Run the script you want:
	- MNIST (with JSON config): `scripts/run_mnist.py --config configs/mnist.json`
	- ImageNette: `scripts/run_imagenette.py`

The script downloads ImageNette automatically on first run.

## Output artifacts
MNIST results are written under `results/mnist_<timestamp>/` (configurable). Example files include:
- `client_class_distribution.png`: class distribution per client
- `lejepa_reconstructions.png`: JEPA gradient inversion reconstructions
- `mae_reconstructions.png`: MAE gradient inversion reconstructions
- `training_loss_curve.png`: LeJEPA/MAE training loss over rounds
- `linear_probe_curve.png`: LeJEPA/MAE probe accuracy over rounds
- `loss_components_log.csv`: per-round loss components for each client

## Project layout
- `src/lejepa_privacy/experiments/mnist_experiment/`: MNIST pipeline modules (data/models/training/privacy/utils).
- `src/lejepa_privacy/experiments/mnist.py`: MNIST entrypoint.
- `src/lejepa_privacy/experiments/imagenette.py`: ImageNette pipeline (refactored).
- `scripts/`: convenience entrypoints.

## Notes
- Adjust settings in `configs/mnist.json` (or pass a custom config path).
- The pipeline uses CUDA automatically when available and falls back to CPU otherwise.
- For reproducibility, the seed and deterministic settings are configurable in the config file.
