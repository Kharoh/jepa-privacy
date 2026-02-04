"""Run the MNIST federated privacy experiment."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from lejepa_privacy.experiments.mnist import run  # type: ignore[import-not-found]


if __name__ == "__main__":
    run()
