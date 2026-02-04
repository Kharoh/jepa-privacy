import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from lejepa_privacy.experiments.mnist_experiment.augment import ViewAugmenter
from lejepa_privacy.experiments.mnist_experiment.data import normalize_mnist, denormalize_mnist
from lejepa_privacy.experiments.mnist_experiment.metrics import bootstrap_ci, permutation_test


class TestMNISTUtilities(unittest.TestCase):
    def test_normalize_roundtrip(self):
        x = torch.rand(4, 28 * 28)
        mean, std = 0.1307, 0.3081
        norm = normalize_mnist(x, mean, std)
        denorm = denormalize_mnist(norm, mean, std)
        self.assertTrue(torch.allclose(x, denorm, atol=1e-5))

    def test_view_augmenter_shape(self):
        x = torch.rand(2, 28 * 28)
        augmenter = ViewAugmenter(
            num_views=3,
            image_shape=(1, 28, 28),
            device="cpu",
            mask_ratio=0.1,
            noise_std=0.0,
            rotation_deg=0.0,
            translation_px=0,
            scale_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
            brightness_range=(1.0, 1.0),
            blur_prob=0.0,
            perspective_prob=0.0,
            solarize_prob=0.0,
            mask_mode="patch",
            patch_size=4,
        )
        views = augmenter(x)
        self.assertEqual(views.shape, (2, 3, 28 * 28))

    def test_stats_helpers(self):
        a = [0.1, 0.2, 0.3]
        b = [0.15, 0.25, 0.35]
        ci = bootstrap_ci(a, num_samples=200)
        p_val = permutation_test(a, b, num_samples=200)
        self.assertEqual(len(ci), 2)
        self.assertTrue(0.0 <= p_val <= 1.0)


if __name__ == "__main__":
    unittest.main()
