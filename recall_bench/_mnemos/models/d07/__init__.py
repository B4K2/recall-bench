"""D07: KDA + per-k-mer fast-weight memory (D06) interleaved with windowed local attention (D02).

Single change from D06: every (hybrid_ratio + 1)-th block's mixer is a causal windowed
local-attention layer (models/shared/local_attention.py) instead of a KDA+k-mer layer. The
window is O(N * window_size) in time with a fixed-size KV ring of `window_size` tokens, so the
arm stays linear in N -- unlike the retired full-softmax hybrid. Exact recall is free inside
the window; beyond it the arm is back on the KDA state + k-mer memory. Intended to be trained
with the sequence-length curriculum (training/curriculum.py); the d06-with-curriculum run is
the control that separates the window's effect from the curriculum's.
"""
from recall_bench._mnemos.models.d05.gpt import KmerGPT
from recall_bench._mnemos.models.d06.attention import FastWeightKmerAttention
from recall_bench._mnemos.models.shared.hybrid import interleave_layer_types
from recall_bench._mnemos.models.shared.local_attention import WindowedLocalAttention


def build_model(config: dict) -> KmerGPT:
    layer_classes = interleave_layer_types(
        config["num_layers"], FastWeightKmerAttention, WindowedLocalAttention,
        int(config.get("hybrid_ratio", 4)))
    return KmerGPT(config, attn_cls=layer_classes)


__all__ = ["build_model"]
