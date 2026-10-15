"""Where data, weights and the model registry live.

Everything the benchmark downloads goes under one home directory:
    $RECALL_BENCH_HOME            (default ~/.cache/recall_bench)
        data/hg38_test.npz        held-out hg38 background   (recall-bench prepare dna)
        data/wikitext103_test.txt text background             (recall-bench prepare text)
        checkpoints/<name>/       registry models             (recall-bench download <name>)
The repo's scripts set RECALL_BENCH_HOME to the repo root, so data/ and checkpoints/ there are used.
"""
import os
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parent


def home() -> Path:
    return Path(os.environ.get("RECALL_BENCH_HOME") or Path.home() / ".cache" / "recall_bench")


def data_dir() -> Path:
    return home() / "data"


def model_dir(name: str) -> Path:
    return home() / "checkpoints" / name


def registry() -> dict:
    return yaml.safe_load((PKG / "models.yaml").read_text())
