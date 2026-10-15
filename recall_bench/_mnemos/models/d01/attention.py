"""Kimi Delta Attention (KDA) — gated delta-rule linear attention.

Per-head recurrent state S in R^{d_k x d_v}, updated per token (decay applied before the
delta-rule write, i.e. the prediction/write use the already-decayed state):
    S_t^decay = diag(alpha_t) @ S_{t-1}
    S_t = S_t^decay + beta_t * k_t (v_t - (S_t^decay)^T k_t)^T
    o_t = S_t^T q_t

alpha_t (channel-wise forget gate) and beta_t (write strength) are input-dependent, per-head
(alpha: per-channel) values in (0, 1) via sigmoid. q/k get a short causal depthwise conv then are
l2-normalized before the recurrence, matching the Kimi Linear / Gated DeltaNet formulation
(arXiv 2510.26692).

Two execution paths, both implementing the exact same recurrence above:
  - fused (default on CUDA): `fla.ops.kda.chunk_kda`, the chunked Triton kernel from the
    flash-linear-attention library (fla-org), which is what makes the 256K-context training
    target in CLAUDE.md §4 tractable — a plain per-token Python loop cannot get there.
  - reference: plain sequential scan in PyTorch. Used on CPU and as the ground truth that the
    fused path is checked against (see tests/test_kda_equivalence.py) — never hand-derive a
    "fast" gated chunked kernel from scratch here; use the maintained, correct one instead.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from recall_bench._mnemos.models.shared import lowmem


class ShortCausalConv(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim, kernel_size, groups=dim, padding=kernel_size - 1, bias=True
        )

    def forward(self, x):
        # x: (batch, seq, dim)
        x = x.transpose(1, 2)
        x = self.conv(x)[..., : x.shape[-1]]
        return x.transpose(1, 2)


class KDAAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        d_model = config["d_model"]
        num_heads = config["num_heads"]
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_fused_kernel = config.get("kda_use_fused_kernel", True)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        conv_kernel = config.get("kda_conv_kernel", 4)
        # Tokens of left context a chunk must be fed (and then discard) for its convolutions to
        # match a whole-sequence pass. Read by the block-level chunked path in models/shared/gpt.py.
        self.left_context = conv_kernel - 1
        self.q_conv = ShortCausalConv(d_model, conv_kernel)
        self.k_conv = ShortCausalConv(d_model, conv_kernel)
        self.v_conv = ShortCausalConv(d_model, conv_kernel)

        # --- Forget gate ------------------------------------------------------------------
        # log(alpha_t) = -exp(A_log) * softplus(alpha_proj(x) + dt_bias), the parameterization
        # used by fla.layers.kda and Mamba, NOT a plain sigmoid.
        #
        # This is not cosmetic. With `alpha = sigmoid(Linear(x))` the gate starts at 0.5, i.e. a
        # one-token state half-life: after 64 tokens the state is scaled by 0.5^64 ~ 5e-20, so
        # there is no surviving signal for a long-range gradient to flow back through, and the
        # model has to climb through a saturating sigmoid to escape. Measured on the d01_rope
        # checkpoint trained with the old sigmoid gate, the *learned* median half-life was 1.48
        # tokens and only 0.96% of channel-tokens exceeded 64bp — the recurrent state was a ~2bp
        # smoother, which is why the §6.4 needle probe sat at chance even at its d=64 positive
        # control. dt_bias is initialized Mamba-style so decay starts near 1 instead.
        self.alpha_proj = nn.Linear(d_model, d_model, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads).uniform_(1, 16)))
        dt_min, dt_max, dt_init_floor = 1e-3, 1e-1, 1e-4
        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp_min(dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        # Weight decay on either of these pulls the gate back toward fast forgetting, which is
        # the failure mode above. scripts/train.py:build_optimizer reads this flag.
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True

        self.beta_proj = nn.Linear(d_model, num_heads)  # per-head write strength

    def _qkv_alpha_beta(self, x, pos_scheme, pos_offset: int = 0, drop_lead: int = 0):
        """`drop_lead` tokens of left context are fed through the causal convs and then
        discarded, so a chunk's conv output matches what a whole-sequence pass would produce;
        `pos_offset` is the absolute index of the first kept token, for the RoPE arms."""
        H, Dh = self.num_heads, self.head_dim

        q = self.q_conv(self.q_proj(x))
        k = self.k_conv(self.k_proj(x))
        v = self.v_conv(self.v_proj(x))
        if drop_lead:
            q, k, v = q[:, drop_lead:], k[:, drop_lead:], v[:, drop_lead:]
            x = x[:, drop_lead:]

        batch, seq_len, d_model = x.shape

        def split_heads(t):
            return t.view(batch, seq_len, H, Dh).transpose(1, 2)  # (B, H, T, Dh)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        if pos_scheme is not None:
            q, k = pos_scheme(q, k, positions=pos_offset)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # log-space decay, computed in fp32: g is always < 0, so alpha = exp(g) is in (0, 1)
        # by construction with no sigmoid saturation to climb through.
        dt = F.softplus(self.alpha_proj(x).float() + self.dt_bias.float())
        g = -torch.exp(self.A_log.float()).repeat_interleave(Dh)[None, None, :] * dt
        g = g.view(batch, seq_len, H, Dh).transpose(1, 2)  # (B, H, T, Dh)

        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)  # (B, H, T)
        return q, k, v, g, beta

    def _forward_reference(self, x, pos_scheme):
        """Plain sequential scan. O(T) Python-level steps — correctness reference / CPU
        fallback only, not for real training runs at long context."""
        q, k, v, g, beta = self._qkv_alpha_beta(x, pos_scheme)
        alpha = torch.exp(g).to(q.dtype)  # g is log-space decay; the recurrence below wants alpha
        batch, H, seq_len, Dh = q.shape

        state = q.new_zeros(batch, H, Dh, Dh)  # S: (B, H, d_k, d_v)
        outputs = []
        for t in range(seq_len):
            q_t, k_t, v_t = q[:, :, t], k[:, :, t], v[:, :, t]
            alpha_t = alpha[:, :, t]  # (B, H, Dh)
            beta_t = beta[:, :, t]  # (B, H)

            state = alpha_t.unsqueeze(-1) * state  # decay first, then delta-rule write
            pred_v = torch.einsum("bhd,bhdv->bhv", k_t, state)
            delta = beta_t.unsqueeze(-1) * (v_t - pred_v)
            state = state + torch.einsum("bhd,bhv->bhdv", k_t, delta)
            outputs.append(torch.einsum("bhd,bhdv->bhv", q_t, state))

        out = torch.stack(outputs, dim=2)  # (B, H, T, Dh)
        return out.transpose(1, 2).reshape(batch, seq_len, H * Dh)

    def _kernel(self, x, pos_scheme, pos_offset, drop_lead, initial_state, output_final_state):
        from fla.ops.kda import chunk_kda

        q, k, v, g, beta = self._qkv_alpha_beta(x, pos_scheme, pos_offset, drop_lead)
        # The Triton kernel requires q/k/v/g/beta to all share one dtype; under autocast,
        # reduction ops (normalize, log) upcast to fp32 while matmuls stay bf16, so force it.
        kernel_dtype = torch.bfloat16 if x.is_cuda else torch.float32
        # fla wants (B, T, H, K) layout and the forget gate in log space.
        q = q.transpose(1, 2).to(kernel_dtype)
        k = k.transpose(1, 2).to(kernel_dtype)
        v = v.transpose(1, 2).to(kernel_dtype)
        g = g.transpose(1, 2).to(kernel_dtype)  # (B, T, H, Dh), already log-space decay
        beta = beta.transpose(1, 2).to(kernel_dtype)  # (B, T, H)

        out, final_state = chunk_kda(
            q, k, v, g, beta,
            scale=1.0,  # our reference recurrence applies no 1/sqrt(d) score scaling
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False,  # already normalized above
            use_gate_in_kernel=False,       # g is already the log-space decay
            use_beta_sigmoid_in_kernel=False,  # beta already post-sigmoid
        )
        return out.reshape(out.shape[0], out.shape[1], -1), final_state

    def _forward_fused(self, x, pos_scheme):
        out, _ = self._kernel(x, pos_scheme, 0, 0, None, False)
        return out

    def _forward_fused_chunked(self, x, pos_scheme, chunk: int):
        """Same recurrence as _forward_fused, evaluated `chunk` tokens at a time with the
        recurrent state carried across boundaries — bounds the per-layer working set so
        contexts that OOM in one pass become reachable. Lossless, not a sliding window."""
        batch, seq_len, d_model = x.shape
        lead = self.q_conv.conv.kernel_size[0] - 1  # conv receptive field, re-fed per chunk

        out_full = None
        state = None
        for start in range(0, seq_len, chunk):
            stop = min(start + chunk, seq_len)
            ctx_start = max(0, start - lead)
            piece, state = self._kernel(
                x[:, ctx_start:stop], pos_scheme,
                pos_offset=start,
                drop_lead=start - ctx_start,
                initial_state=state,
                output_final_state=True,
            )
            if out_full is None:
                # Allocated on first use so the output dtype matches the kernel's.
                out_full = piece.new_empty(batch, seq_len, piece.shape[-1])
            out_full[:, start:stop] = piece
        return out_full

    def forward_chunked(self, x_ext, pos_scheme, state, pos_offset: int, drop_lead: int):
        """One chunk of the recurrence, resuming from `state` and returning the state after it.

        `x_ext` carries `drop_lead` tokens of left context ahead of the chunk proper (for the
        causal convs); `pos_offset` is the absolute index of the first kept token. Presence of
        this method is what marks an attention as supporting the block-level chunked path — the
        windowed and full-attention variants do not implement it and fall back to whole-sequence
        execution.
        """
        out, final_state = self._kernel(
            x_ext, pos_scheme, pos_offset, drop_lead,
            initial_state=state, output_final_state=True)
        return self.o_proj(out), final_state

    def forward(self, x, pos_scheme=None):
        if self.use_fused_kernel and x.is_cuda:
            chunk = lowmem.chunk_size()
            if chunk is not None and x.shape[1] > chunk:
                out = self._forward_fused_chunked(x, pos_scheme, chunk)
            else:
                out = self._forward_fused(x, pos_scheme)
        else:
            out = self._forward_reference(x, pos_scheme)
        return self.o_proj(out)
