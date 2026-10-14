"""Zero-shot long-range recall benchmark for genomic (and, as a control, text) LMs."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def registry() -> dict:
    return yaml.safe_load((ROOT / "models.yaml").read_text())


def model_dir(name: str) -> Path:
    return ROOT / "checkpoints" / name
