"""KDA + discrete k-mer addressed memory (D05).

The KDA recurrence of models/d01 is unchanged (global compressive state S). Its per-token output
h_t (pre o_proj) is the write content/gate input for one or more k-mer tables (models/d05/
kmer_memory.py, one per k in `kmer_mem_hierarchical_ks` or the single `kmer_mem_k`), and the
gated read of the slot addressed by the k-mer ending at t is added back to h_t before o_proj:
    o_t = o_proj( h_t + sum_k sigmoid(lambda_k) * W_r^k M^k[a^k_t] )
With `kmer_mem_slots_per_kmer > 1` each M^k[a] is an S-way set read/written through a learned
router (see kmer_memory.py); the router's load-balance loss is drained by models/d05/gpt.py.
Reads see the slot as of the motif's *previous* occurrence (causal); the write at t goes to the
slot of the k-mer ending at t-1 (see kmer_memory.py docstring for why the one-position offset).

Addresses come from raw token ids, which the attention layer never sees, so models/d05/gpt.py
sets `self.kmer_addr` ((B, T) per k, full-sequence) before running the blocks; the chunked
inference path slices it by `pos_offset`. Table state is carried across chunks alongside S.
"""
import torch

from recall_bench._mnemos.models.d01.attention import KDAAttention
from recall_bench._mnemos.models.d05.kmer_memory import KmerMemory


class KmerMemAttention(KDAAttention):
    def __init__(self, config: dict):
        super().__init__(config)
        d_model = config["d_model"]
        ks = list(config.get("kmer_mem_hierarchical_ks") or []) or [int(config.get("kmer_mem_k", 8))]
        if config.get("kmer_mem_update", "gated_ema") != "gated_ema":
            raise NotImplementedError("only kmer_mem_update: gated_ema is implemented")
        if not config.get("kmer_mem_reset_per_sequence", True):
            raise NotImplementedError("table is always reset per sequence")
        self.ks = ks
        self.memories = torch.nn.ModuleList([
            KmerMemory(k, d_model, int(config.get("kmer_mem_dim", 32)),
                       float(config.get("kmer_mem_gate_init", -4.0)),
                       int(config.get("kmer_mem_scan_chunk", 64)),
                       write_gate=str(config.get("kmer_mem_write_gate", "content")),
                       surprise_init=float(config.get("kmer_mem_surprise_init", 2.0)),
                       slots_per_kmer=int(config.get("kmer_mem_slots_per_kmer", 1)),
                       router_temp=float(config.get("kmer_mem_router_temp", 1.0)),
                       router_mode=str(config.get("kmer_mem_router_mode", "soft")),
                       write_gate_init=config.get("kmer_mem_write_gate_init"))
            for k in ks])
        self.kmer_addr = None   # dict k -> ((B,T) read addr, (B,T) write addr), set by KmerGPT per forward
        self.enabled = True     # False on layers excluded by kmer_mem_layers

    def _addr(self, k, T, pos_offset):
        r, w = self.kmer_addr[k]   # full-sequence (read, write) address pair
        return r[:, pos_offset:pos_offset + T], w[:, pos_offset:pos_offset + T]

    def _apply_memory(self, h, tables, pos_offset):
        """h (B, T, d_model). tables: list per k or None."""
        if not self.enabled:
            return h, tables
        B, T, _ = h.shape
        out, new_tables = h, []
        for i, (k, mem) in enumerate(zip(self.ks, self.memories)):
            tab = tables[i] if tables is not None else None
            r, tab = mem(h, self._addr(k, T, pos_offset), tab)
            out = out + r
            new_tables.append(tab)
        return out, new_tables

    def forward(self, x, pos_scheme=None):
        assert self.kmer_addr is not None, "KmerGPT must set kmer_addr before the blocks run"
        if self.use_fused_kernel and x.is_cuda:
            h, _ = self._kernel(x, pos_scheme, 0, 0, initial_state=None, output_final_state=False)
        else:
            h = self._forward_reference_pre_proj(x, pos_scheme)
        h, _ = self._apply_memory(h, None, 0)
        return self.o_proj(h)

    def _forward_reference_pre_proj(self, x, pos_scheme):
        q, k, v, g, beta = self._qkv_alpha_beta(x, pos_scheme)
        alpha = torch.exp(g).to(q.dtype)
        B, H, T, Dh = q.shape
        state = q.new_zeros(B, H, Dh, Dh)
        outs = []
        for t in range(T):
            state = alpha[:, :, t].unsqueeze(-1) * state
            pred = torch.einsum("bhd,bhdv->bhv", k[:, :, t], state)
            state = state + torch.einsum("bhd,bhv->bhdv", k[:, :, t],
                                         beta[:, :, t].unsqueeze(-1) * (v[:, :, t] - pred))
            outs.append(torch.einsum("bhd,bhdv->bhv", q[:, :, t], state))
        return torch.stack(outs, dim=2).transpose(1, 2).reshape(B, T, H * Dh)

    def forward_chunked(self, x_ext, pos_scheme, state, pos_offset: int, drop_lead: int):
        S, tables = state if state is not None else (None, None)
        h, S_new = self._kernel(x_ext, pos_scheme, pos_offset, drop_lead,
                                initial_state=S, output_final_state=True)
        h, tables = self._apply_memory(h, tables, pos_offset)
        return self.o_proj(h), (S_new, tables)
