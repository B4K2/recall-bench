"""Per-k-mer fast-weight (delta-rule) memory -- D06.

d05 stores one *vector* per k-mer and overwrites it with a gated EMA. Its needle cliff sits at
D/4^k = 1 (measured 2026-08-19): every further occurrence of the same 8-mer at an unrelated locus
is a genuine overwrite (the surprise signal is ~1 at chain rank 1), so across a gap of D bases the
slot is clobbered ~D/4^k times. d05c tried S discrete sub-slots behind a softmax router and the
router collapsed. D06 keeps the exact k-mer address but replaces the vector slot by a tiny
fast-weight *matrix* written with the delta rule under a context key:

    slot a :  M_a in R^{d_v x d_k}                       (d05: m_a in R^{d})
    write  :  M_a <- M_a (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
              k_t = l2norm(W_k h_t), v_t = W_v h_t, beta_t = sigmoid(W_b h_t + b)
    read   :  o_t = M_a q_t,  q_t = l2norm(W_q h_t)

Two writes of the same k-mer with (near-)orthogonal context keys k_1 _|_ k_2 no longer clobber
each other: M holds v_1 k_1^T + v_2 k_2^T and the query k_1 recovers v_1. Capacity per slot is
~d_k roughly-orthogonal contexts (the delta-rule Prop. 2 bound, applied per address), so the
total capacity goes from 4^k items to ~4^k * d_k and the overwrite cliff should move from
D ~ 4^k to D ~ 4^k * d_k. That is the falsifiable claim: the needle copy-advantage cliff of d05
(at ~65K for k=8) should move right by roughly d_k. If it does not, keys are not reproducible /
discriminative enough across occurrences and the table idea is exhausted.

This is d05c with the discrete router replaced by continuous associative keys: no router, no
load-balance loss, one read + one write event per position (same event stream as d05; d05c needed
2*S*T). Read and write addresses keep d05's one-position offset (write h_t into the slot of the
k-mer ending at t-1 -> induction lookup at the next occurrence). With d_k = 1 and k = q = 1 this
is exactly d05's EMA, so d05 is the d_k = 1 corner of this family.

Chunked algorithm. As in d05, all events on one address form a causal chain and distinct chains
are independent, so events are stable-sorted by (batch, address), each chain cut into pieces of
`scan_chunk` events, and the pieces stacked. Inside a piece the delta rule is solved in closed form
with the WY representation (DeltaNet chunkwise form, Yang et al. 2024): for keys K, values V,
gates beta (padding rows have beta = 0 and contribute nothing),
    T = (I + stril(diag(beta) K K^T))^{-1} diag(beta)          (C x C, unit-lower-tri solve)
    W = T K,  U = T V                                           pseudo-keys / pseudo-values
    prod_i (I - beta_i k_i k_i^T) = I - W^T K                   whole-piece transition  P
    S_C = S_0 P + U^T K                                         whole-piece input       Bp
    o_n = S_0 q_n + sum_{j<n} (q_n^T k_j) (u_j - S_0 w_j)     intra-piece outputs
Pieces of one chain are carried with a loop over max(pieces per chain) steps of
carry <- carry @ P + Bp, one batched matmul over all chains per step (short even at 262144: the
average k=8 chain has ~4 occurrences, i.e. one piece). Read events have beta = 0 (no update) and
carry q; write events carry (k, v, beta) and a zero q. Differentiable end to end with plain
autograd; the table is only materialised when a caller needs to carry it across chunks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


class FastWeightKmerMemory(nn.Module):
    """One k-mer table of d_v x d_k fast-weight matrices plus its read/write heads. Exposes the
    attribute contract KmerGPT / KmerMemAttention rely on (read_gate, write_gate, aux_loss,
    router_stats, k, scan_chunk)."""

    def __init__(self, k: int, d_in: int, d_k: int, d_v: int, gate_init: float, scan_chunk: int,
                 write_gate_init: float | None = None, n_slots: int | None = None):
        super().__init__()
        self.k, self.d_k, self.d_v, self.scan_chunk = k, d_k, d_v, scan_chunk
        self.n_addr = 4 ** k
        self.n_slots = self.n_addr if n_slots is None else int(n_slots)   # 1 for address_mode shared
        self.q_proj = nn.Linear(d_in, d_k, bias=False)
        self.k_proj = nn.Linear(d_in, d_k, bias=False)
        self.v_proj = nn.Linear(d_in, d_v, bias=False)
        self.gate_init = float(gate_init)                                  # read_gate init
        self.write_gate_init = float(gate_init if write_gate_init is None else write_gate_init)
        self.write_gate = nn.Linear(d_in, 1)               # beta_t logit; re-applied by KmerGPT
        nn.init.constant_(self.write_gate.bias, self.write_gate_init)
        self.write_gate.bias._no_weight_decay = True
        self.read_proj = nn.Linear(d_v, d_in, bias=False)
        self.read_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.read_gate._no_weight_decay = True
        self.aux_loss = None        # no router -> nothing to balance (KmerGPT contract)
        self.router_stats = None

    # ---- state --------------------------------------------------------------------------------

    def empty_table(self, batch: int, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_zeros(batch, self.n_slots, self.d_v, self.d_k, dtype=torch.float32)

    # ---- the segmented chunked delta-rule scan ------------------------------------------------

    def scan(self, addr, q, k, v, beta, table, return_table=True):
        """addr (B,E) long (-1 = no-op event), q/k (B,E,d_k) fp32, v (B,E,d_v) fp32,
        beta (B,E) fp32 in [0,1), table (B,V,d_v,d_k) fp32.
        Returns outputs (B,E,d_v) = M_{addr} q *before* the event's own write, and the table after
        all writes (None when return_table is False)."""
        B, E, dk = q.shape
        dv = v.shape[-1]
        V, C = self.n_slots, self.scan_chunk
        dev = q.device
        valid = addr >= 0
        key = (torch.arange(B, device=dev)[:, None] * V + addr.clamp_min(0)).masked_fill(~valid, -1)
        flat_key, flat_valid = key.reshape(-1), valid.reshape(-1)
        order = torch.argsort(flat_key, stable=True)
        n_inv = int((~flat_valid).sum().item())
        order = order[n_inv:]
        n = order.numel()
        outs = v.new_zeros(B * E, dv)
        if n == 0:
            return outs.view(B, E, dv), (table if return_table else None)

        skey = flat_key[order]
        seg_start = torch.ones(n, dtype=torch.bool, device=dev)
        seg_start[1:] = skey[1:] != skey[:-1]
        seg_id = torch.cumsum(seg_start.long(), 0) - 1
        n_seg = int(seg_id[-1].item()) + 1
        first = torch.nonzero(seg_start).squeeze(1)
        rank = torch.arange(n, device=dev) - first[seg_id]
        piece_in_seg, pos_in_piece = rank // C, rank % C
        seg_pieces = torch.zeros(n_seg, dtype=torch.long, device=dev)
        seg_pieces.scatter_reduce_(0, seg_id, piece_in_seg + 1, reduce="amax")
        piece_base = torch.cumsum(seg_pieces, 0) - seg_pieces
        piece_id = piece_base[seg_id] + piece_in_seg
        n_pieces = int(seg_pieces.sum().item())
        max_pieces = int(seg_pieces.max().item())

        # --- pack events into (n_pieces, C, .) ; padding rows are all-zero (beta 0 -> no-op) ---
        def pack(x):
            out = x.new_zeros((n_pieces, C) + x.shape[2:])
            out[piece_id, pos_in_piece] = x.reshape((B * E,) + x.shape[2:])[order]
            return out
        Q, K, Vv, Bt = pack(q), pack(k), pack(v), pack(beta)          # (P,C,dk) (P,C,dk) (P,C,dv) (P,C)

        # --- WY representation inside a piece ---
        KK = torch.einsum("pnd,pmd->pnm", K, K)                        # (P,C,C)
        tri = torch.ones(C, C, dtype=torch.bool, device=dev).tril(-1)
        A = torch.eye(C, device=dev, dtype=K.dtype) + (Bt[:, :, None] * KK).masked_fill(~tri, 0.0)
        Tm = torch.linalg.solve_triangular(A, torch.diag_embed(Bt), upper=False, unitriangular=True)
        W = torch.bmm(Tm, K)                                             # (P,C,dk)
        U = torch.bmm(Tm, Vv)                                            # (P,C,dv)
        P_piece = torch.eye(dk, device=dev, dtype=K.dtype) - torch.einsum("pnd,pne->pde", W, K)  # (P,dk,dk)
        B_piece = torch.einsum("pnv,pnd->pvd", U, K)                     # (P,dv,dk)

        # --- carry the state across the pieces of each chain ---
        seg_key = skey[first]
        seg_b, seg_addr = seg_key // V, seg_key % V
        carry = table[seg_b, seg_addr]                                   # (n_seg, dv, dk)
        init = v.new_zeros(n_pieces, dv, dk)
        for j in range(max_pieces):
            has = seg_pieces > j
            pid = (piece_base + j).clamp_max(n_pieces - 1)
            init[pid[has]] = carry[has]
            nxt = torch.bmm(carry, P_piece[pid]) + B_piece[pid]
            carry = torch.where(has[:, None, None], nxt, carry)

        # --- intra-piece outputs: o_n = S_0 q_n + sum_{j<n} (q_n.k_j)(u_j - S_0 w_j) ---
        QK = torch.einsum("pnd,pmd->pnm", Q, K).masked_fill(~tri, 0.0)   # (P,C(n),C(j))
        S0q = torch.einsum("pvd,pnd->pnv", init, Q)
        S0w = torch.einsum("pvd,pnd->pnv", init, W)
        O = S0q + torch.bmm(QK, U - S0w)                                 # (P,C,dv)
        outs[order] = O[piece_id, pos_in_piece]
        new_table = table.index_put((seg_b, seg_addr), carry) if return_table else None
        return outs.view(B, E, dv), new_table

    def _heads(self, h):
        q = F.normalize(self.q_proj(h).float(), dim=-1)
        k = F.normalize(self.k_proj(h).float(), dim=-1)
        v = self.v_proj(h).float()
        beta = torch.sigmoid(self.write_gate(h).float().squeeze(-1))
        return q, k, v, beta

    def _events(self, read_addr, write_addr, q, k, v, beta):
        """Interleave per position: read event (beta 0, q) then write event (k, v, beta, q = 0)."""
        B, T, _ = q.shape
        addr = torch.stack([read_addr, write_addr], 2).reshape(B, 2 * T)
        q2 = torch.stack([q, torch.zeros_like(q)], 2).reshape(B, 2 * T, -1)
        k2 = torch.stack([torch.zeros_like(k), k], 2).reshape(B, 2 * T, -1)
        v2 = torch.stack([torch.zeros_like(v), v], 2).reshape(B, 2 * T, -1)
        b2 = torch.stack([torch.zeros_like(beta), beta], 2).reshape(B, 2 * T)
        return addr, q2, k2, v2, b2

    def _scan_rw(self, read_addr, write_addr, q, k, v, beta, table, return_table):
        B, T, _ = q.shape
        with torch.autocast(device_type=q.device.type, enabled=False):   # fp32 recurrence
                ev, table = self.scan(*self._events(read_addr, write_addr, q, k, v, beta), table,
                                  return_table=return_table)
        return ev.view(B, T, 2, -1)[:, :, 0], table

    def forward(self, h, addr, table, return_table=True):
        """h (B,T,d_in); addr = (read_addr, write_addr) each (B,T) long; table (B,V,d_v,d_k) or
        None. Returns (contribution to the mixer output (B,T,d_in), new table or None)."""
        read_addr, write_addr = addr
        B = h.shape[0]
        if table is None:
            table = self.empty_table(B, h)
        q, k, v, beta = self._heads(h)
        if torch.is_grad_enabled():
            reads, table = torch.utils.checkpoint.checkpoint(
                self._scan_rw, read_addr, write_addr, q, k, v, beta, table, return_table,
                use_reentrant=False)
        else:
            reads, table = self._scan_rw(read_addr, write_addr, q, k, v, beta, table, return_table)
        out = self.read_proj(reads.to(h.dtype)) * torch.sigmoid(self.read_gate.float()).to(h.dtype)
        return out, table

    # ---- sequential ground truth (tests) -----------------------------------------------------

    def reference(self, h, addr, table=None):
        read_addr, write_addr = addr
        B, T, _ = h.shape
        base = table if table is not None else self.empty_table(B, h)
        q, k, v, beta = self._heads(h)
        slots, reads = {}, []
        get = lambda b, a: slots.get((b, a), base[b, a])
        for b in range(B):
            for t in range(T):
                a = int(read_addr[b, t])
                reads.append(get(b, a) @ q[b, t] if a >= 0 else v.new_zeros(self.d_v))
                w = int(write_addr[b, t])
                if w >= 0:
                    M = get(b, w)
                    slots[(b, w)] = M + beta[b, t] * torch.outer(v[b, t] - M @ k[b, t], k[b, t])
        new_table = base.clone()
        for (b, a), M in slots.items():
            new_table[b, a] = M.detach()
        reads = torch.stack(reads).view(B, T, self.d_v)
        out = self.read_proj(reads.to(h.dtype)) * torch.sigmoid(self.read_gate.float()).to(h.dtype)
        return out, new_table
