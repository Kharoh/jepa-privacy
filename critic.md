# Critique of the MNIST Federated Privacy Experiment (LeJEPA vs MAE)

This critique focuses on the MNIST federated-learning privacy experiment that compares LeJEPA and MAE in `src/lejepa_privacy/experiments/mnist_experiment/`, driven by `scripts/run_mnist.py` and `configs/mnist.json`. The current pipeline is a strong starting point for a comparative study, but there are important methodological, privacy, and reproducibility issues that limit the validity of any conclusions about privacy in federated learning.

## High‑level summary
- The experiment does **not match the federated threat model** it claims to study. It uses raw gradients for attacks even though the server only receives model updates in FedAvg. This makes the leakage analysis optimistic (and inconsistent with actual data sharing).
- LeJEPA and MAE are **not compared under matched conditions** (augmentation pipelines, masking, objectives, architecture capacity, and compute differ), so privacy differences may be confounded.
- The privacy evaluation is **single‑client, single‑batch, single‑seed** and uses **augmented views rather than raw data**. That makes the reported privacy metrics fragile and hard to interpret.
- The DP configuration is **not a valid DP‑FL mechanism** (no per‑sample clipping, no accountant, and noise added to vectors that are not shared), so privacy claims cannot be supported.

## Strengths worth keeping
- Clear modularization of data, models, privacy attack, and training logic.
- Consistent logging and checkpointing via `utils.py`.
- Multiple privacy metrics (MSE/PSNR/cosine/relative L2/correlation) and visualization support.
- Baseline reconstruction metrics are provided (random and mean baselines).

## Major methodological issues

### 1) Threat‑model mismatch (critical)
---
- In FL, the server typically receives **model updates/weights**, not **per‑batch gradients**. Your attack uses `update["gradients"]` from the client, which are not transmitted by the current FL protocol in `training.py`. This breaks the threat model.
---

- If you want to study **server‑side gradient inversion**, the system should explicitly send gradients (or be framed as federated SGD). If you want **FedAvg** (which is what you implement), the attack must be on **model updates** (e.g., DLG-style reconstruction from weight deltas).

### 2) Model/attack mismatch (critical)
- The inversion attack uses the **global model** inside `GradientInversionAttack`, but the gradients were computed from the **local model** after a local training step. This mismatch makes the inversion invalid or at least inconsistent.
- The gradient is computed on the local model, then the local model is updated, then the **global model** aggregates and advances. The attacker uses the current global model but tries to match gradients computed from a different model instance.

### 3) Not a “foundation model” comparison
- LeJEPA and MAE here are **small, from‑scratch models** (ViT tiny / small MLP‑like heads) trained on MNIST. The experiment does not reflect typical “foundation model” settings (large‑scale pretraining, transfer learning, frozen encoders).
- Privacy claims about *foundation models* are therefore not supported.

## Experimental design and fairness gaps

---
### 1) Non‑matched augmentations
- LeJEPA uses a heavy multi‑view augmentation pipeline (`ViewAugmenter`), while MAE uses **identity augmentation by default**.
- This creates **distribution shift** in both the training objective and the privacy attack target. JEPA gradients and reconstructions are based on augmented views; MAE gradients are based on near‑raw data.
- The result: privacy metrics compare **different inputs**, not model behaviors under comparable data.
---

### 2) Objective mismatch and masking
- MAE uses masking in the reconstruction objective, but JEPA does not use masking (unless explicitly set in the augmenter). These are **not equivalent objectives**, and the privacy leakage may be driven by different parts of the input space.
- The reported privacy metrics are about reconstructions of **augmented inputs** for JEPA, but about **original inputs** for MAE. This is not an apples‑to‑apples privacy metric.

### 3) Capacity/compute imbalance
- LeJEPA and MAE parameter counts, outputs, and compute are not normalized. A larger model or decoder can amplify gradient signal and leakage.
- The experiment uses a fixed learning rate (1e‑4), and no effort is made to equalize convergence speed or training difficulty.

## Federated learning setup issues
### 1) FedAvg weighting is incorrect for non‑IID splits
- Aggregation uses simple mean across clients, not weighted by **number of samples per client**.
- With the Dirichlet split, clients have **different dataset sizes**, making the aggregate biased.

### 2) Non‑IID sampling is not controlled
- `load_mnist_non_iid` does not enforce **equal data sizes** across clients. This complicates fairness and interpretability.
- There is no reporting of **effective per‑client dataset sizes** or entropy of label distributions beyond a plot.

### 3) No client subsampling, no realistic FL dynamics
- All clients participate every round, and each runs identical epochs. This does not represent common FL regimes where only a subset participates each round.
- There’s no notion of dropped clients, stragglers, or variable batch sizes.

## Privacy evaluation flaws
### 1) Single client, single batch evaluation
- Privacy metrics are computed only for `client_idx = 0` and only from **the last local training batch**.
- This ignores variability across clients, batches, and data classes. It also overfits to a single data point per evaluation round.

---
### 2) Reconstruction target is not the raw data
- For LeJEPA, `x_test_views` is used for attack and metrics, which are **augmented views**, not the original raw images. That breaks the privacy interpretation.
- For MAE, the recon target is typically the raw data (or masked reconstruction target), so you are comparing **different targets**.
---

---
### 3) Inconsistent data normalization/denormalization
- Reconstruction metrics denormalize using MNIST mean/std, but augmented views may not preserve valid MNIST pixel distributions.
- Added noise and aggressive augmentations produce pixel values outside $[0, 1]$, making PSNR/MSE less meaningful.
---

---
### 4) No attack baselines beyond DLG‑style
- You only include a gradient inversion attack variant with cosine/MSE loss. There is no comparison with stronger or alternative attacks (e.g., iDLG, RLB, or analytic inversion for linear probes).
- There is no **attack success rate** or per‑class analysis beyond plotting snapshots.
---

## Differential privacy is not a valid DP‑FL implementation
- The DP configuration adds noise to **whole‑model gradients** or **update vectors**, not to **per‑sample gradients**. This is not a standard DP mechanism and does not provide a meaningful $(\varepsilon, \delta)$ guarantee.
- There is no accounting (e.g., RDP accountant), so privacy budgets are unknown.
- Noise is applied to vectors **not actually shared** (the gradients are not transmitted), so the DP defense is not protecting the real attack surface.

## Statistical analysis and reporting gaps
- Statistical summaries use a small number of evaluation points (every 250 rounds), which is **too few for meaningful CIs**.
- The bootstrapped CIs and permutation tests treat **rounds as independent samples**, which is incorrect because rounds are highly correlated.
- No multiple‑seed runs. With only one seed, you cannot distinguish signal from randomness.

## Reproducibility and configurability issues
- The experiment uses random augmentations for JEPA without fixed RNG per view, making results **non‑deterministic** even with seeds.
- `configs/mnist.json` lists a large number of knobs, but they are not tied to a standardized experiment protocol or sweep.
---
- The README focuses on ImageNette and the MNIST pipeline is under‑documented, which is confusing for the MNIST privacy study.
----
## Interpretation risks and missing comparisons
- There is no comparison to **non‑self‑supervised baselines** (e.g., supervised learning, SimCLR‑style), so it is unclear whether privacy behavior is specific to JEPA/MAE or generic to representation learning.
---
- There is no measure of **utility vs privacy trade‑off** (e.g., probe accuracy vs privacy metrics) on the same axis.
---
---
- The experiment lacks a **communication cost** or **training time** comparison; compute differences might drive privacy differences.
---

## Concrete improvement recommendations
### Fix the threat model
- Decide between **FedAvg (model update sharing)** vs **federated SGD (gradient sharing)** and align the attack accordingly.
- If FedAvg: store and attack **model deltas** or **server‑observed updates**.

### Make LeJEPA vs MAE comparable
- Harmonize augmentations: either apply the same augmentations to both, or use shared perturbation policies.
- Control model capacity and compute: match parameter counts, or use the same backbone with different heads/objectives.
- Align reconstruction targets: either attack the original raw images in both cases or always attack augmented views (but then interpret privacy as leakage of augmented data only).

### Strengthen privacy evaluation
- Evaluate **multiple clients and multiple batches** per round, and report distributions (mean/median/quantiles).
- Use multiple random seeds and report variance.
- Include at least one stronger or alternative inversion baseline and possibly membership inference.

### Make DP claims defensible
- Implement **per‑sample gradient clipping** and **noise addition** on the *shared* signal.
- Add a DP accountant and report $(\varepsilon, \delta)$.

### Improve statistical validity
- Use independent trials (multi‑seed) for CIs.
- Use a mixed‑effects model or per‑round paired analysis rather than treating rounds as IID.

---
### Document the exact protocol
- Clarify whether the attack is on gradients or updates.
- Specify exactly which data are used for reconstruction (raw vs augmented vs normalized).
- Provide reproducibility notes for RNGs and random augmentations.
---

## Bottom line
Right now, the experiment is **useful for quick qualitative intuition** but **not reliable for privacy conclusions** in federated learning. The main blockers are the threat‑model mismatch, non‑matched inputs/objectives between LeJEPA and MAE, and the weak statistical design. Fixing these will make the comparison credible and allow you to argue meaningful privacy differences.
