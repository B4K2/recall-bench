"""Standard causal softmax attention (D03/D06's hybrid partner to KDA). Uses SDPA, which
dispatches to a flash-attention backend, so this is efficient at the "small constant" ratio the
hybrid arms use it at (see CLAUDE.md §3) — not meant to run as every layer at 256K context."""
import torch.nn as nn
import torch.nn.functional as F


class FullAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        d_model = config["d_model"]
        num_heads = config["num_heads"]
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, pos_scheme=None):
        batch, seq_len, d_model = x.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(x).view(batch, seq_len, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, H, Dh).transpose(1, 2)

        if pos_scheme is not None:
            q, k = pos_scheme(q, k)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.o_proj(out)
