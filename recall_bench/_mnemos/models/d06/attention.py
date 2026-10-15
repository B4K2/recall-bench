"""KDA + per-k-mer fast-weight (delta-rule) memory (D06).

Identical to models/d05/attention.py except that each k-mer table holds d_v x d_k fast-weight
matrices (models/d06/fastweight_memory.py) instead of d-vectors. Single-axis change from d05:
same KDA backbone, same addresses (incl. the one-position read/write offset), same read gate and
fusion point (pre o_proj). During the full-sequence training forward the table is not
materialised (only chained carries are computed); the chunked inference path carries it.
"""
import torch

from recall_bench._mnemos.models.d05.attention import KmerMemAttention
from recall_bench._mnemos.models.d06.fastweight_memory import FastWeightKmerMemory


class FastWeightKmerAttention(KmerMemAttention):
    def __init__(self, config: dict):
        super().__init__(config)
        d_model = config["d_model"]
        self.memories = torch.nn.ModuleList([
            FastWeightKmerMemory(k, d_model,
                                 int(config.get("kmer_fw_dk", 16)), int(config.get("kmer_fw_dv", 16)),
                                 float(config.get("kmer_mem_gate_init", -1.0)),
                                 int(config.get("kmer_mem_scan_chunk", 16)),
                                 write_gate_init=config.get("kmer_mem_write_gate_init"),
                                 n_slots=1 if config.get("kmer_mem_address_mode") == "shared" else None)
            for k in self.ks])

    def _apply_memory(self, h, tables, pos_offset, return_table=True):
        if not self.enabled:
            return h, tables
        B, T, _ = h.shape
        out, new_tables = h, []
        for i, (k, mem) in enumerate(zip(self.ks, self.memories)):
            tab = tables[i] if tables is not None else None
            r, tab = mem(h, self._addr(k, T, pos_offset), tab, return_table=return_table)
            out = out + r
            new_tables.append(tab)
        return out, new_tables

    def forward(self, x, pos_scheme=None):
        assert self.kmer_addr is not None, "KmerGPT must set kmer_addr before the blocks run"
        if self.use_fused_kernel and x.is_cuda:
            h, _ = self._kernel(x, pos_scheme, 0, 0, initial_state=None, output_final_state=False)
        else:
            h = self._forward_reference_pre_proj(x, pos_scheme)
        h, _ = self._apply_memory(h, None, 0, return_table=False)
        return self.o_proj(h)
