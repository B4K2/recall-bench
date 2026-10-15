"""Positional schemes. NoPE arms (D01-D03, D07) pass through unchanged; NTK-scaled RoPE
(D04-D06) is implemented once here so it stays a single-axis ablation (see CLAUDE.md §5)."""
import torch


class NoPositionalEncoding:
    def __call__(self, q, k, positions=None):
        return q, k


class NTKScaledRoPE:
    """NTK-aware RoPE interpolation. Rescales the rotation base so long-range/low-freq
    components stretch to cover `target_length` while short-range/high-freq components stay
    close to behavior at `trained_length`. Formula per the NTK-aware scaling scheme (Peng et al.
    / community "NTK-RoPE"): base' = base * (target_length / trained_length) ** (dim / (dim - 2)).
    """

    def __init__(self, dim: int, base: float = 10000.0, trained_length: int = 262144,
                 target_length: int | None = None):
        self.dim = dim
        self.trained_length = trained_length
        self.target_length = target_length or trained_length
        scale = max(self.target_length / self.trained_length, 1.0)
        adjusted_base = base * (scale ** (dim / (dim - 2)))
        inv_freq = 1.0 / (adjusted_base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer_inv_freq = inv_freq

    def _cos_sin(self, seq_len: int, device, dtype, offset: int = 0):
        t = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.register_buffer_inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def __call__(self, q, k, positions=None):
        # q, k: (batch, heads, seq, head_dim)
        # `positions` is the absolute index of q[..., 0, :]. It is 0 for a whole-sequence
        # forward, but non-zero when the sequence is processed in chunks (see
        # models/shared/lowmem.py) — without honouring it every chunk would be rotated as if
        # it started at position 0, which is silently wrong rather than merely approximate.
        seq_len = q.shape[-2]
        cos, sin = self._cos_sin(seq_len, q.device, q.dtype, offset=int(positions or 0))
        cos, sin = cos[None, None], sin[None, None]
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


def build_pos_scheme(config: dict, head_dim: int):
    pos_type = config.get("pos_type", "none")
    if pos_type == "none":
        return NoPositionalEncoding()
    if pos_type == "ntk_rope":
        return NTKScaledRoPE(
            dim=head_dim,
            base=config.get("rope_base", 10000.0),
            trained_length=config.get("context_length", 262144),
            target_length=config.get("rope_target_length"),
        )
    raise ValueError(f"unknown pos_type: {pos_type}")
