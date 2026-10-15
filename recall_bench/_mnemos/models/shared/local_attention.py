"""Causal windowed local attention (D02/D05's hybrid partner to KDA).

Blocked sliding-window implementation (Mistral-style): the sequence is split into blocks of
`window_size`, and each query block attends to itself (causally) and the immediately preceding
block. This gives an effective causal window between `window_size` and `2*window_size - 1`
tokens, and costs O(T * window_size) rather than O(T^2) — an exact fixed-W window would need a
ragged/banded attention pattern that plain SDPA can't express efficiently.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowedLocalAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        d_model = config["d_model"]
        num_heads = config["num_heads"]
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = config.get("window_size", 512)
        # For the block-level chunked path (GPT._encode_chunked): the previous chunk's last
        # `left_context` tokens are prepended so every kept position sees its full window.
        self.left_context = self.window_size

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward_chunked(self, x_ext, pos_scheme, state, pos_offset: int, drop_lead: int):
        """One chunk with `drop_lead` (= 0 or window_size) tokens of left context. Exact
        whenever the chunk grid is aligned with the block grid, i.e. chunk starts are multiples
        of window_size -- then each kept block's "previous block" is exactly the carried one.
        The layer keeps no recurrent state (the carry *is* the KV ring), so state is None."""
        W = self.window_size
        assert drop_lead in (0, W) and (pos_offset - drop_lead) % W == 0, (
            f"chunked windowed attention needs window-aligned chunks "
            f"(pos_offset={pos_offset}, drop_lead={drop_lead}, window={W})")
        out = self._attend(x_ext, pos_scheme, positions=pos_offset - drop_lead)
        return out[:, drop_lead:], None

    def forward(self, x, pos_scheme=None):
        return self._attend(x, pos_scheme, positions=0)

    def _attend(self, x, pos_scheme, positions: int):
        batch, seq_len, d_model = x.shape
        H, Dh, W = self.num_heads, self.head_dim, self.window_size

        pad = (-seq_len) % W
        x_padded = F.pad(x, (0, 0, 0, pad)) if pad else x
        T = x_padded.shape[1]
        num_blocks = T // W

        q = self.q_proj(x_padded).view(batch, T, H, Dh).transpose(1, 2)
        k = self.k_proj(x_padded).view(batch, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x_padded).view(batch, T, H, Dh).transpose(1, 2)

        if pos_scheme is not None:
            q, k = pos_scheme(q, k, positions or None)

        def to_blocks(t):
            return t.view(batch, H, num_blocks, W, Dh)

        q_blk, k_blk, v_blk = to_blocks(q), to_blocks(k), to_blocks(v)

        # shift blocks by one so block i's "previous" is block i-1 (zeros for block 0)
        prev_k = F.pad(k_blk, (0, 0, 0, 0, 1, 0))[:, :, :-1]
        prev_v = F.pad(v_blk, (0, 0, 0, 0, 1, 0))[:, :, :-1]

        kv_k = torch.cat([prev_k, k_blk], dim=3)  # (B, H, num_blocks, 2W, Dh)
        kv_v = torch.cat([prev_v, v_blk], dim=3)

        device = x.device
        local_i = torch.arange(W, device=device)
        local_j = torch.arange(2 * W, device=device)
        causal_mask = local_j.unsqueeze(0) <= (local_i.unsqueeze(1) + W)  # (W, 2W)
        block0_mask = causal_mask.clone()
        block0_mask[:, :W] = False  # block 0 has no valid previous block

        full_mask = causal_mask.unsqueeze(0).expand(num_blocks, -1, -1).clone()
        if num_blocks > 0:
            full_mask[0] = block0_mask
        attn_bias = torch.zeros(num_blocks, W, 2 * W, device=device, dtype=q.dtype)
        attn_bias.masked_fill_(~full_mask, torch.finfo(q.dtype).min)

        def merge(t):
            return t.permute(0, 2, 1, 3, 4).reshape(batch * num_blocks, H, t.shape[3], Dh)

        q_m, k_m, v_m = merge(q_blk), merge(kv_k), merge(kv_v)
        bias_m = attn_bias.unsqueeze(0).expand(batch, -1, -1, -1).reshape(batch * num_blocks, 1, W, 2 * W)

        out = F.scaled_dot_product_attention(q_m, k_m, v_m, attn_mask=bias_m)
        out = out.view(batch, num_blocks, H, W, Dh).permute(0, 2, 1, 3, 4).reshape(batch, H, T, Dh)
        out = out.transpose(1, 2).reshape(batch, T, d_model)[:, :seq_len]
        return self.o_proj(out)
