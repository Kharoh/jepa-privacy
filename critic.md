Critique of `src/lejepa_privacy/experiments/mnist.py`

This document is an exhaustive critique of the MNIST experiment implemented in `src/lejepa_privacy/experiments/mnist.py`. It covers high-level experimental design, federated learning setup, model and training choices, augmentation/data pipeline, privacy metrics and attacks, evaluation fairness, software engineering quality, reproducibility, and suggestions for improvements.

## 1. High-level experimental design issues

- The script attempts to do too many things at once: representation learning (LeJEPA vs MAE), federated training, privacy measurement (MI estimator present but unused), gradient inversion attacks, t-SNE visualizations, and linear probe downstream evaluation. This mix dilutes the experimental focus and makes it hard to draw causal conclusions.
- The research question is vague. There's no clear single hypothesis being tested with controlled variables. Is the claim "LeJEPA provides better privacy than MAE"? If so, what exactly is being measured as privacy?
- Variables are not isolated. The code changes multiple factors between methods (architecture, objective, number of views, augmentation pipeline). Any measured differences could come from any of these axes.
- No formal DP accounting. A `DPConfig` object exists, but there's no calculation of privacy budget (epsilon/delta) or use of an accountant (Moments Accountant, RDP, etc.). The code currently only applies clipping and Gaussian noise without reporting the privacy loss.

## 2. Federated learning methodology

- Full participation every round. The code aggregates updates from all clients each round. Real federated settings typically sample a subset of clients each round; full participation reduces realism and hides variance introduced by client sampling.
- Single-client privacy evaluation. Gradients and reconstructions for privacy attacks are taken only from `client_idx = 0`, making the reported privacy leakage non-representative.
- **Gradients for attacks do not correspond cleanly to the published training updates. The code computes gradients on a test batch after local training and uses those for inversion; therefore the attack targets a different gradient signal than the one sent to the server.**
- **Small and inconsistent local training. Local training uses 2 epochs over a batch sampled from client's data (batch size capped at 32). It’s unclear how many unique samples are seen; this makes local updates unstable.**
- Aggregation ignores weighting by local dataset size. The server averages parameters equally across clients, even though client dataset sizes may differ, which biases global updates.

## 3. Model architecture & training choices

- ViT backbone usage is questionable. The `ViTLeJEPAEncoder` and `MAEViT` reuse `vit_tiny_patch16_224` with img_size overloaded. ViT pretraining assumptions (patch size, embedding dimension, positional encodings) may not be valid for $32\times32$ MNIST images.
- Baseline mismatch: MAE vs LeJEPA architectures differ in more than loss. MAE has decoder and a reconstruction target; LeJEPA uses an invariance loss + SIGReg without decoder. Comparing them is not controlled for capacity, decoder presence, or objective hardness.
- **No consistent normalization. MNIST images are converted to tensors but not normalized to zero mean and unit variance; ViTs and many models are sensitive to input scale.**
- SIGReg’s random projection inside forward pass introduces uncontrolled stochasticity and may vary across calls and devices. No seeded generator is used for the random projection within SIGReg.
- The invariance loss is simple mean-square deviation from per-sample mean across views; it lacks redundancy reduction or joint embedding decorrelation terms (e.g., Barlow Twins, VICReg, SimSiam stop-gradient trick), so it may converge to trivial solutions.
- Hyperparameters are arbitrary and not justified (e.g., LAMB = 0.005, NUM_ROUNDS = 10000). The scale of experiments is not explained.

## 4. Augmentation & data pipeline

- Augmentations are aggressive for MNIST. Rotation of 25°, translation 4 px on 32×32 images, scaling up to 1.2 and down to 0.8, blur, brightness/contrast jitter, noise — together these likely distort digit semantics and make the training signal inconsistent with MNIST labels.
- **Augmentations are used for MAE too, which is a reconstruction objective; aggressive transforms will make reconstruction harder and change the baseline behaviour.**
- **The `ViewAugmenter` applies masking per pixel, not per patch. MAE typically uses patch masking; mixing pixel masking with patch masking introduces a mismatch between objectives.**
- Data reshaping is messy. The code uses flattened images, sometimes (B, n, D) views, sometimes (B, 1, D) and sometimes (B, C, H, W). That complexity invites shape bugs.

## 5. Privacy metrics, attacks, and interpretation

- The Gaussian MI estimator exists but is never used to fill `results["*"]["mi"]`. The experiments’ reported MI numbers will be zero or missing; the printed summary is misleading.
- The MINE estimator is included but not used in the main flow. It’s expensive and requires careful training/validation to be reliable.
- Gradient inversion attack setup has issues:
  - The attack optimizes dummy inputs to match gradients from the model’s loss computed on dummy inputs vs true gradient. But the model’s loss differs (LeJEPA invariance vs MAE reconstruction); comparing attacks across objectives is tricky.
  - The attack uses a cosine similarity loss by default, and later reports cosine similarity metric as a privacy indicator — this is circular.
  - TV regularization weight is fixed to 1e-4 without ablation; total variation interacts with pixel scaling and input range.
- **Metrics like PSNR and cosine similarity are reported without confidence intervals or baselines (random reconstruction, aggregate mean). No statistical testing.**
- The printed “Improvement” calculations are suspect: some denominators may be zero or near-zero and there’s no guardrails against division by zero.

## 6. Evaluation fairness and experimental controls

- LeJEPA uses `NUM_VIEWS=4` by default; MAE clients are constructed with `num_views=1`. Multi‑view learning itself offers privacy amplification by mixing signals; to fairly compare you should either give both models the same number of views or explain why differing views are an intended part of the comparison.
- Model parameter counts and capacity aren’t matched. ViT vs MLP encoder vs CNN encoder differences will affect leakage, and must be controlled.
- Attack hyperparameters are identical for both models, which is fair in one sense, but the attack difficulty is model/objective dependent; it might be necessary to tune attack hyperparameters per target to ensure a robust comparison.

## 7. Software engineering & reproducibility

- **Everything is in a single ~2240-line script. That’s unmaintainable and hard to test. Split into modules (`data`, `models`, `training`, `privacy`, `utils`) and import them.**
- **Hard-coded configuration inside `run_federated_privacy_experiment()`. Use a config file (YAML/JSON/argparse) and/or Hydra to sweep experiments and record parameters.**
- **Randomness is not fully controlled. You seed Torch and numpy, but many randoms use default RNGs (transformations, timm internals, torch.randperm in DataLoader usage, TSNE). For reproducibility, seed all places and optionally set deterministic flags for CUDA.**
- **No unit tests and no lightweight smoke tests (e.g., run 1 round, 1 client, tiny data) before running full experiments.**
- **Logging: output goes to stdout and a CSV in the cwd. Use structured logging and a dedicated `logs/` or `results/` directory. Overwriting files can occur across runs.**
- **No checkpoints or resume support. Long experiments are fragile.**

## 8. Performance and computational cost

- NUM_ROUNDS = 10000 and expensive gradient inversion (600 iterations) means the run will take infeasible time/GPU. This is not practical for development or experiments.
- Frequent plotting (t-SNE, reconstructions) and heavy operations in the training loop are expensive and can cause IO bottlenecks.
- TSNE in the training loop is not appropriate; t-SNE is only useful offline on carefully subsampled data.

## 9. Metrics and plotting issues

- The code uses `plot_metric_curve` for plotting losses but mixes incompatible series (inv vs sigreg) as if they were comparable magnitudes and meanings.
- Plotting at every `PLOT_ROUNDS` may overwrite previous artifacts because filenames are not versioned or timestamped.
- No figure seeding or dpi control that’s consistent across environments.

## 10. Statistical & scientific validity

- No repeated runs or seeds. Single run results are noisy and non‑conclusive.
- **No hypothesis testing. Claims about privacy improvement are unquantified (no p-values, no effect sizes with uncertainty).**
- No sanity baselines like random gradients, zeroed gradients, or explicit leakage bounds.

---

# Concrete suggestions to improve the experiment

1. Separate concerns. Move models, data loading, attacks, and the federated training loop into separate modules. Keep `run_*` functions minimal and controlled by a config.
2. Define a clear hypothesis and match the experiment to it. For example: "Given equal architecture and input views, LeJEPA reduces gradient inversion reconstruction quality compared to MAE".
3. Control architectures and capacity. Compare models with the same encoder capacity and same number of views. If MAE requires a decoder, keep decoder small or add a similar head to LeJEPA to match parameter count.
4. Use proper DP accounting if you claim DP benefits. Integrate an RDP accountant (e.g., Opacus, or implement RDP composition) and report epsilon/delta.
5. Make federated protocol realistic: sample clients each round, weight aggregation by local dataset size, and run experiments with different participation rates.
6. Run multiple random seeds (≥3, preferably 5–10) and report means and confidence intervals for privacy metrics.
7. Reduce computational load for development: start with NUM_ROUNDS=50 and fewer attack iterations, then scale up for final runs.
8. For MI estimation, either fully integrate Gaussian estimator usage or remove it. If using MINE, isolate its training and validation carefully and report variance.
9. Add unit tests: small tests for `ViewAugmenter`, `SIGReg`, `apply_dp_to_tensors`, and the inversion attack (sanity checks on shape, decreasing loss during attack).
10. Improve logging: structured JSON logs with timestamps, separate `results/` directory, and checkpointing of models.

---

If you want, I can implement a refactor: a slim reproducible baseline experiment with matched architectures, a small federated simulation, and a robust gradient inversion evaluation pipeline. Tell me which direction you'd prefer (quick refactor to run on CPU, or full-scale experiment with DP accounting and multiple seeds).
