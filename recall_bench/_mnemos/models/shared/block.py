import torch.nn as nn

from recall_bench._mnemos.models.shared.ffn import SwiGLU
from recall_bench._mnemos.models.shared.norm import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-norm residual block. The sequence-mixing operator is injected so that adding a
    new attention variant means adding one attention.py, not editing shared code."""

    def __init__(self, attention_cls, config: dict):
        super().__init__()
        d_model = config["d_model"]
        d_ffn = config.get("d_ffn", 4 * d_model)
        self.attn_norm = RMSNorm(d_model)
        self.attn = attention_cls(config)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ffn)

    def forward(self, x, pos_scheme=None):
        x = x + self.attn(self.attn_norm(x), pos_scheme=pos_scheme)
        x = x + self.ffn(self.ffn_norm(x))
        return x
