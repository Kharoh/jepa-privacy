# LeJEPA vs MAE Federated Privacy (MNIST)

This experiment compares LeJEPA and MAE in a federated learning setup and evaluates privacy leakage using mutual information and gradient inversion attacks. It uses MNIST and creates non-IID client splits with a Dirichlet class distribution. A per-client class distribution plot is saved to disk.

## What’s inside
- MNIST non-IID client split with Dirichlet class proportions
- Federated learning loop with LeJEPA and MAE baselines
- Mutual information and gradient inversion privacy metrics
- Saved plot: `client_class_distribution.png`

## How to run
1. Install dependencies.
2. Run `main.py`.

The script downloads MNIST to the local `data/` folder on first run.

## Output artifacts
- `client_class_distribution.png`: heatmap of class counts per client
- `lejepa_reconstructions.png`: JEPA gradient inversion reconstructions
- `mae_reconstructions.png`: MAE gradient inversion reconstructions
- `lejepa_recon_steps_round{N}.png`: JEPA reconstruction steps for classes at round N
- `mae_recon_steps_round{N}.png`: MAE reconstruction steps for classes at round N
- `lejepa_tsne_round{N}_val.png`: LeJEPA validation t-SNE (non-augmented samples)
- `mae_tsne_round{N}_val.png`: MAE validation t-SNE (non-augmented samples)
- `training_loss_curve.png`: JEPA/MAE training loss over rounds
- `linear_probe_curve.png`: JEPA/MAE linear probe accuracy over rounds
- `loss_components_log.csv`: per-round loss components for each client and global model

## Notes
- You can tune the non-IID severity with `DIRICHLET_ALPHA` in `main.py`.
- For faster runs, reduce `NUM_ROUNDS`, `SAMPLES_PER_CLIENT`, or probe sample sizes.
- LeJEPA views are now generated via small random crops (resized back to $28\times28$).
- The script uses CUDA automatically when available and falls back to CPU otherwise.
- Toggle CNN backbones via `USE_CNN` in `run_federated_privacy_experiment()` (default: `True`).
- LeJEPA uses multi-augmentation views (affine + brightness/contrast + blur + masking + noise) in `ViewAugmenter`.

## Differential privacy mode
You can enable DP-style clipping + Gaussian noise in federated updates and shared gradients.
Edit the DP configuration block in `run_federated_privacy_experiment()`:
- `DP_ENABLED`: toggle DP mode
- `DP_CLIP_NORM`: $L_2$ norm clip for updates/gradients
- `DP_NOISE_MULTIPLIER`: noise scale ($\sigma$) applied as `noise_multiplier * clip_norm`

When DP is enabled, client updates are clipped and noised before aggregation, and the
gradients used for privacy analysis are similarly noised.
