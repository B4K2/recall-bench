"""GPT assembler for D05: computes k-mer addresses from the token ids once per forward and
hands them to every KmerMemAttention before the blocks run (the blocks only see hidden states)."""
import torch

from recall_bench._mnemos.models.d05.attention import KmerMemAttention
from recall_bench._mnemos.models.d05.kmer_memory import kmer_addresses, remap_addresses, shift_write_addresses
from recall_bench._mnemos.models.shared.gpt import GPT


def _layer_mask(spec, n_layers):
    if spec in (None, "all"):
        return [True] * n_layers
    if spec == "last":
        return [i == n_layers - 1 for i in range(n_layers)]
    idxs = set(int(i) for i in spec)
    return [i in idxs for i in range(n_layers)]


class KmerGPT(GPT):
    def __init__(self, config: dict, attn_cls=KmerMemAttention):
        # attn_cls is KmerMemAttention / a subclass, or a per-layer list mixing those with
        # non-k-mer mixers (D07 interleaves WindowedLocalAttention). Only the k-mer layers
        # (those with a .memories attribute) take part in the .enabled/.ks/.kmer_addr contract.
        super().__init__(attn_cls, config)
        assert self._kmer_attns(), "KmerGPT needs at least one k-mer memory layer"
        for blk, on in zip(self.blocks, _layer_mask(config.get("kmer_mem_layers", "all"),
                                                     config["num_layers"])):
            if hasattr(blk.attn, "memories"):
                blk.attn.enabled = on
        # GPT.__init__ ends with self.apply(_init_weights), which zeroes every nn.Linear bias --
        # including each memory's write_gate.bias, silently discarding kmer_mem_gate_init (read_gate
        # is a bare Parameter, so it survived; that asymmetry is how this was found: sigmoid(bias)
        # logged 0.5 at step 1 instead of sigmoid(gate_init)). Restore it after the sweep.
        for attn in self._kmer_attns():
            for mem in attn.memories:
                torch.nn.init.constant_(mem.write_gate.bias, mem.write_gate_init)
        self.ks = self._kmer_attns()[0].ks
        self.address_mode = str(config.get("kmer_mem_address_mode", "kmer"))
        remap_addresses(torch.zeros(1, 1, dtype=torch.long), self.address_mode, 1)  # validate
        self.balance_coef = float(config.get("kmer_mem_balance_coef", 0.01))

    def _kmer_attns(self):
        return [blk.attn for blk in self.blocks if hasattr(blk.attn, "memories")]

    def _set_addresses(self, idx):
        with torch.no_grad():
            addr = {}
            for k in self.ks:
                r = remap_addresses(kmer_addresses(idx, k), self.address_mode, 4 ** k)
                addr[k] = (r, shift_write_addresses(r))
        for attn in self._kmer_attns():
            attn.kmer_addr = addr

    def encode(self, idx, layers=None, reduce=None):
        self._set_addresses(idx)
        return super().encode(idx, layers=layers, reduce=reduce)

    def kmer_aux_loss(self):
        """Sum of the set-associative routers' load-balance terms from the last forward (0 when
        slots_per_kmer == 1, where no router exists). Each term is >= 1 and == 1 for uniform
        entry usage, so the constant is subtracted to keep the added loss at 0 for a balanced
        router rather than a fixed offset."""
        terms = [mem.aux_loss for attn in self._kmer_attns() if attn.enabled
                 for mem in attn.memories if mem.aux_loss is not None]
        if not terms:
            return None
        return torch.stack(terms).sum() - float(len(terms))

    def kmer_metrics(self):
        """Scalar diagnostics from the last forward for MLflow: raw balance aux loss (before
        balance_coef), router entropy / occupancy averaged over memories (worst-case occupancy
        max/min), and mean read/write gate openness. {} when there is no k-mer memory."""
        out = {}
        mems = [mem for attn in self._kmer_attns() if attn.enabled for mem in attn.memories]
        if not mems:
            return out
        aux = self.kmer_aux_loss()
        if aux is not None:
            out["kmer_aux_loss"] = aux.item()
        stats = [m.router_stats for m in mems if m.router_stats is not None]
        if stats:
            out["router_entropy"] = sum(s["router_entropy"] for s in stats) / len(stats)
            out["router_occ_max"] = max(s["router_occ_max"] for s in stats)
            out["router_occ_min"] = min(s["router_occ_min"] for s in stats)
        with torch.no_grad():
            out["read_gate_mean"] = torch.stack(
                [torch.sigmoid(m.read_gate.float()).mean() for m in mems]).mean().item()
            out["write_gate_bias_mean"] = torch.stack(
                [torch.sigmoid(m.write_gate.bias.float()).mean() for m in mems]).mean().item()
        return out

    def forward(self, idx, targets=None):
        logits, loss = super().forward(idx, targets)
        aux = self.kmer_aux_loss()
        if loss is not None and aux is not None and self.balance_coef:
            loss = loss + self.balance_coef * aux
        return logits, loss


def build_model(config: dict) -> KmerGPT:
    return KmerGPT(config)
