"""D01-new: same pure-KDA arm as models/d01, at ~2M params (see configs/d01_new.yaml).

Only the config differs (d_model/num_layers/num_heads), so the builder is D01's — scripts/train.py
resolves the builder by model_id, which is why this module exists at all.
"""
from recall_bench._mnemos.models.d01 import build_model

__all__ = ["build_model"]
