import torch
import torch.nn as nn
import torch.utils.checkpoint

from recall_bench._mnemos.models.shared import lowmem
from recall_bench._mnemos.models.shared.block import TransformerBlock
from recall_bench._mnemos.models.shared.norm import RMSNorm
from recall_bench._mnemos.models.shared.pos import build_pos_scheme


class GPT(nn.Module):
    """Shared GPT assembler: embedding -> N x TransformerBlock -> final norm -> lm_head.
    Weight-tied embedding/lm_head. `attention_cls` is the only per-arm injection point."""

    def __init__(self, attention_cls, config: dict):
        """`attention_cls` is either a single class (used for every layer) or a list of
        per-layer classes of length config["num_layers"] (see models/shared/hybrid.py)."""
        super().__init__()
        self.config = config
        d_model = config["d_model"]
        vocab_size = config["vocab_size"]

        if isinstance(attention_cls, (list, tuple)):
            assert len(attention_cls) == config["num_layers"]
            layer_classes = attention_cls
        else:
            layer_classes = [attention_cls] * config["num_layers"]

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cls, config) for cls in layer_classes]
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        head_dim = d_model // config["num_heads"]
        self.pos_scheme = build_pos_scheme(config, head_dim)
        self.gradient_checkpointing = config.get("gradient_checkpointing", False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _supports_chunked(self) -> bool:
        return all(hasattr(block.attn, "forward_chunked") for block in self.blocks)

    def _encode_chunked(self, idx, chunk: int, layers=None, reduce=None):
        """Whole-block chunked evaluation: for each layer, walk the sequence in chunks carrying
        the attention's recurrent state across boundaries, running norm/FFN/residual on the
        chunk too. Everything except the residual stream itself becomes O(chunk) instead of
        O(T), which is what makes 512K/1M reachable — chunking attention alone leaves the FFN's
        full-length d_ffn intermediates as the binding term.

        Exact, not approximate: norms, FFN and residual adds are position-wise, and the
        attention resumes from the previous chunk's state (see models/shared/lowmem.py).
        Inference only — the in-place residual write below is not autograd-safe, so the caller
        must have grad disabled.
        """
        x = self.tok_emb(idx)
        seq_len = x.shape[1]
        captured = []

        for li, block in enumerate(self.blocks):
            state = None
            lead = getattr(block.attn, "left_context", 0)
            carry = None  # pre-update x for the last `lead` positions of the previous chunk
            for start in range(0, seq_len, chunk):
                stop = min(start + chunk, seq_len)
                xc = x[:, start:stop]
                # Saved before the in-place write below overwrites these positions, since the
                # next chunk's convolutions need the *pre-update* values.
                tail = xc[:, stop - start - lead:].clone() if lead else None

                if carry is not None:
                    x_ext, drop_lead = torch.cat([carry, xc], dim=1), carry.shape[1]
                else:
                    x_ext, drop_lead = xc, 0

                attn_out, state = block.attn.forward_chunked(
                    block.attn_norm(x_ext), self.pos_scheme, state, start, drop_lead)
                y = xc + attn_out
                y = y + block.ffn(block.ffn_norm(y))
                x[:, start:stop] = y
                carry = tail
            if layers is not None and li in layers:
                # x is fully updated for this layer here; reduce reads it, never keeps it.
                captured.append(reduce(self.final_norm(x) if li == len(self.blocks) - 1 else x))
        if layers is not None:
            return captured
        return self.final_norm(x)

    def _resolve_layers(self, layers):
        n = len(self.blocks)
        out = set()
        for l in layers:
            l = l + n if l < 0 else l
            assert 0 <= l < n, f"layer index {l} out of range for {n} blocks"
            out.add(l)
        return out

    def encode(self, idx, layers=None, reduce=None):
        """Hidden states after the final norm, before lm_head — the representation the
        downstream linear probes in benchmarks/ pool over.

        `layers`: optional block indices (negative allowed). When given, returns a list with
        one entry per requested layer (ascending index order): the residual stream after that
        block, with the final norm applied to the last block's output only. `reduce` (default
        identity) is applied to each captured tensor immediately — pass the pooling there so
        the chunked long-context path never holds more than one layer's full hidden state.
        """
        if layers is not None:
            layers = self._resolve_layers(layers)
            reduce = reduce or (lambda h: h)
        chunk = lowmem.chunk_size()
        if (chunk is not None and idx.shape[1] > chunk and not torch.is_grad_enabled()
                and self._supports_chunked()):
            return self._encode_chunked(idx, chunk, layers, reduce)

        x = self.tok_emb(idx)
        captured = []
        for li, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, self.pos_scheme, use_reentrant=False
                )
            else:
                x = block(x, pos_scheme=self.pos_scheme)
            if layers is not None and li in layers:
                captured.append(reduce(self.final_norm(x) if li == len(self.blocks) - 1 else x))
        if layers is not None:
            return captured
        return self.final_norm(x)

    def forward(self, idx, targets=None):
        x = self.encode(idx)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n
