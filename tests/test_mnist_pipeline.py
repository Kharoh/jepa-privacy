import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from lejepa_privacy.experiments.mnist_experiment.config import ExperimentConfig
from lejepa_privacy.experiments.mnist_experiment.privacy import (
    GradientInversionAttack,
    UpdateInversionAttack,
)
from lejepa_privacy.experiments.mnist_experiment.training import run_federated_privacy_experiment
from lejepa_privacy.experiments.mnist_experiment.utils import setup_logging


def _fast_attack(self, original_data, true_grad, **kwargs):
    if isinstance(original_data, torch.Tensor):
        return original_data.detach().cpu().clone(), {}
    return torch.zeros((1, 1), dtype=torch.float32), {}


class TestMNISTPipeline(unittest.TestCase):
    def test_small_pipeline_runs(self):
        config = ExperimentConfig(
            seed=123,
            deterministic=True,
            device="cpu",
            input_dim=28 * 28,
            emb_dim=16,
            proj_dim=8,
            num_clients=2,
            samples_per_client=20,
            dirichlet_alpha=0.5,
            num_rounds=1,
            num_views=2,
            lamb=0.1,
            use_cnn=False,
            use_vit=False,
            image_shape=(1, 28, 28),
            eval_every=1,
            plot_rounds=[],
            plot_classes=[0, 1],
            plot_steps=[0, 5],
            batch_size=8,
            local_epochs=1,
            max_batches_per_epoch=1,
            normalize_mean=0.1307,
            normalize_std=0.3081,
            augmenter_kwargs={
                "mask_ratio": 0.0,
                "noise_std": 0.0,
                "rotation_deg": 0.0,
                "translation_px": 0,
                "scale_range": (1.0, 1.0),
                "contrast_range": (1.0, 1.0),
                "brightness_range": (1.0, 1.0),
                "blur_prob": 0.0,
                "perspective_prob": 0.0,
                "solarize_prob": 0.0,
                "solarize_threshold": 0.0,
                "mask_mode": "pixel",
                "patch_size": 4,
            },
            mae_augmenter_kwargs={
                "enabled": False,
                "mask_ratio": 0.0,
                "noise_std": 0.0,
                "rotation_deg": 0.0,
                "translation_px": 0,
                "scale_range": (1.0, 1.0),
                "contrast_range": (1.0, 1.0),
                "brightness_range": (1.0, 1.0),
                "blur_prob": 0.0,
                "perspective_prob": 0.0,
                "solarize_prob": 0.0,
                "solarize_threshold": 0.0,
                "mask_mode": "patch",
                "patch_size": 4,
            },
            output_dir="results",
            checkpoint_every=0,
            resume_from=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(tmpdir)
            try:
                with (
                    mock.patch.object(GradientInversionAttack, "attack", new=_fast_attack),
                    mock.patch.object(UpdateInversionAttack, "attack", new=_fast_attack),
                    mock.patch("torch.cuda.is_available", return_value=False),
                ):
                    results = run_federated_privacy_experiment(config, Path(tmpdir), logger)
            finally:
                for handler in list(logger.handlers):
                    handler.close()
                    logger.removeHandler(handler)

        self.assertIn("lejepa", results)
        self.assertIn("mae", results)
        self.assertEqual(len(results["lejepa"]["loss"]), 1)


if __name__ == "__main__":
    unittest.main()
