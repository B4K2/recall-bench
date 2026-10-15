"""Discrete k-mer addressed memory (concept note: discrete_kmer_memory_idea.pdf).

A table M in R^{4^k x d}. At position t, with a_t the *causal* k-mer ending at t:
    r_t        = M[a_t]                            (read: what followed earlier occurrences of a_t)
    gamma_t    = sigmoid(W_g h_t + b_g)            (scalar write gate; b_g init negative -> keep)
    u_t        = W_u h_t                           (write content)
    M[a_{t-1}] <- (1 - gamma_t) * M[a_{t-1}] + gamma_t * u_t
When `write_gate="surprise"` (config `kmer_mem_write_gate`), gamma_t additionally depends on how
badly the slot being written already predicts u_t:
    s_t        = ||u_t - M[a_{t-1}]|| / (||u_t|| + eps)     (scale-free "surprise", detached)
    gamma_t    = sigmoid(W_g h_t + b_g + w_e * s_t)
A slot that already holds the right continuation gets a near-null write; a novel motif writes at
full strength. This matters because the recall cliff of this memory is set by the *overwrite rate*
(a slot is clobbered ~D/4^k times across a gap of D bases), not by mem_dim or by address collisions
-- the addresses are exact. Suppressing redundant writes to a fraction p of positions moves the
cliff from D ~ 4^k to D ~ 4^k / p at no memory cost. Note a *stochastic* keep-probability p would
buy nothing (it drops the target's write too); the selectivity has to be content-driven.
s_t needs the slot content before the write, which the closed-form scan below only produces once
gamma is already known, so this runs the scan twice: pass 1 under gamma_0 = sigmoid(W_g h + b_g)
to read out M[a_{t-1}], then pass 2 with the corrected gamma. Pass 1 is detached -- s_t is a
statistic computed against a reference EMA trace, not an exactly-consistent fixed point, and it
should be described that way in writeup. Gradients still reach w_e (s_t is a constant multiplier)
and reach everything else through pass 2.
For a one-hot address the delta rule M <- M + beta(u - M) is *identical* to the gated EMA with
gamma = beta, so there is no separate "delta" update to implement here; the error-driven part is
exactly and only this gate.

`slots_per_kmer = S > 1` makes each address an S-way *set* rather than a single slot, which is the
fix for the failure mode the write gate above cannot touch. On real hg38 the same k-mer at
unrelated loci carries genuinely different continuations (measured: the surprise signal s_t is
0.975 at chain rank 1, i.e. a repeat almost never predicts its own next write), so every write is
a real overwrite of something equally real -- the address is simply an ambiguous key. S entries per
address plus a router that sends different contexts to different entries buys S-way disambiguation
at cost S, where widening the key to k+j bases would cost 4^j.
    p_t        = softmax(W_p h_t / tau)                     (B, T, S) routing weights
    r_t        = sum_s p_{t,s} M[a_t, s]                    (read: soft over the set)
    M[a_{t-1}, s] <- (1 - gamma_t p_{t,s}) M[..] + gamma_t p_{t,s} u_t
Both read and write route on h_t, so similar contexts land on the same entry -- a learned
locality-sensitive hash over the set. Note the routing is deliberately *state-independent*: the
closed-form scan below needs every gate known before it runs, so a d04/HOLA-style evict-by-
beta*||e|| rule (which inspects the stored entries) would break it. Learned routing is the
tractable stand-in and should be described as such, not as HOLA eviction.
Implementation: the set is folded into the address space (V = 4^k * S, entry index a*S + s) and
each position emits S read events followed by S write events, so `scan` below is reused unchanged
with a 2*S*T event stream. Cost is S x the scan and S x the table. S = 1 is byte-identical to the
single-slot memory (no router is even constructed).
A router can collapse onto one entry, which would silently undo all of this; `load_balance_loss`
(config `kmer_mem_balance_coef`, MoE-style, collected by KmerGPT) counteracts that, and
`router_entropy`/per-entry occupancy should be logged during training to confirm it did.

Read and write addresses are offset by one position on purpose (changed 2026-08-18): h_t is causal,
so writing h_t into the slot of the k-mer that ends at t stores nothing about the base *after* that
k-mer, which is exactly what the next occurrence needs to predict. Writing h_t into the slot of the
k-mer ending at t-1 (the motif that base t followed) makes the read at the next occurrence an
induction lookup. Reads at t happen before the write at t. Positions whose read address is -1
(window contains N/PAD, or t < k-1) do not read; positions whose write address is -1 do not write.

All occurrences of one k-mer form a causal chain and distinct chains are independent, so the
sequence is reorganized by address and scanned as many segmented recurrences (note §7):
  1. stable-sort the (B*T) positions by (batch, address) -> chains are contiguous, in time order
  2. split every chain into pieces of `scan_chunk` consecutive occurrences and stack the pieces
     into a (n_pieces, scan_chunk, d) matrix (padding waste < 1 piece per chain)
  3. inside a piece the EMA is a closed form: r_n = c_0 * prod_{i<n} (1-g_i)
       + sum_{j<n} g_j u_j prod_{j<i<n} (1-g_i), a strictly-lower-triangular
     (scan_chunk x scan_chunk) decay matrix built from pairwise differences of a log cumsum
     (differences before exp -> no overflow; underflow to 0 is the right answer)
  4. carry across the pieces of one chain with a loop over max(pieces per chain) steps —
     each step is one gather + FMA over all chains at once, so it is short even at 262144
     (a chain of L occurrences costs ceil(L/scan_chunk) steps; the ~4-occurrence average chain
     at k=8 costs one).
Differentiable end to end (reads, gates, writes) with plain autograd; the piece matrices are
the only O(T*scan_chunk) intermediate and are rebuilt under checkpointing.
"""
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint

from recall_bench._mnemos.tokenizer import TOKEN_TO_ID

BASE_IDS = torch.tensor([TOKEN_TO_ID[b] for b in "ACGT"])  # A=0, C=1, G=2, T=3 in address space


def kmer_addresses(idx: torch.Tensor, k: int) -> torch.Tensor:
    """(B, T) token ids -> (B, T) int64 address of the k-mer ending at each position
    (base-4, most recent base least significant), or -1 if any base in the window is not ACGT
    or fewer than k bases precede. Causal by construction: uses tokens <= t only."""
    lut = torch.full((max(int(idx.max().item()) + 1, 8),), -1, dtype=torch.long, device=idx.device)
    lut[BASE_IDS.to(idx.device)] = torch.arange(4, device=idx.device)
    base = lut[idx.long()]                     # (B, T) in {-1, 0..3}
    invalid = (base < 0).long()
    # window has any invalid base <=> cumulative invalid count differs across the window
    cs = torch.cumsum(invalid, dim=1)
    cs_shift = torch.nn.functional.pad(cs, (k, 0))[:, :-k]  # count strictly before window
    win_invalid = (cs - cs_shift) > 0
    addr = torch.zeros_like(base)
    b = base.clamp_min(0)
    for i in range(k):
        shifted = torch.nn.functional.pad(b, (i, 0))[:, : b.shape[1]] if i else b
        addr = addr + shifted * (4 ** i)
    addr[win_invalid] = -1
    addr[:, : k - 1] = -1
    return addr


ADDRESS_MODES = ("kmer", "position_hash", "shared")


def remap_addresses(addr: torch.Tensor, mode: str, n_addr: int) -> torch.Tensor:
    """Addressing controls for the review ablation. Every mode keeps the -1 (no-op) positions of
    the k-mer address, so the event count, gates and scan cost are identical to `kmer`:
      kmer          : the k-mer's own address (the arm under test)
      position_hash : a fixed hash of the absolute position -> same table size, but a later
                      occurrence of the same k-mer can no longer find what was written
      shared        : every event hits slot 0 -> one fast-weight matrix, no addressing at all
    A random *permutation* of k-mer addresses is deliberately not a mode: it is a relabelling
    of the slots and is indistinguishable from `kmer`."""
    if mode == "kmer":
        return addr
    valid = addr >= 0
    if mode == "position_hash":
        t = torch.arange(addr.shape[1], device=addr.device, dtype=torch.long)
        # 32-bit integer mixer, not a linear map: t*c mod 4^k is periodic in t and would give an
        # exact position echo at distance 4^k -- i.e. fake recall at that needle distance.
        m = 0xFFFFFFFF
        h = t ^ (t >> 16)
        h = (h * 0x45D9F3B) & m
        h = h ^ (h >> 16)
        h = (h * 0x45D9F3B) & m
        h = (h ^ (h >> 16)) % n_addr
        return torch.where(valid, h.expand_as(addr), addr)
    if mode == "shared":
        return torch.where(valid, torch.zeros_like(addr), addr)
    raise ValueError(f"kmer_mem_address_mode must be one of {ADDRESS_MODES}, got {mode!r}")


def shift_write_addresses(addr: torch.Tensor) -> torch.Tensor:
    """Write address for position t = read address of position t-1 (-1 at t=0). Apply to the
    full-sequence address tensor *before* chunking so chunk boundaries stay consistent."""
    return torch.nn.functional.pad(addr, (1, 0), value=-1)[:, :-1]


class KmerMemory(nn.Module):
    """One k-mer table plus its read/write heads. `forward` runs the segmented scan for a whole
    (chunk of) sequence and returns the read vectors and the updated table."""

    def __init__(self, k: int, d_in: int, mem_dim: int, gate_init: float, scan_chunk: int,
                 write_gate: str = "content", surprise_init: float = 2.0,
                 slots_per_kmer: int = 1, router_temp: float = 1.0,
                 router_mode: str = "soft", write_gate_init: float | None = None):
        super().__init__()
        if slots_per_kmer < 1:
            raise ValueError("slots_per_kmer must be >= 1")
        self.k, self.mem_dim, self.scan_chunk = k, mem_dim, scan_chunk
        self.S, self.router_temp = slots_per_kmer, router_temp
        if router_mode not in ("soft", "top1_ste"):
            raise ValueError(f"router_mode must be soft|top1_ste, got {router_mode!r}")
        self.router_mode = router_mode
        self.n_addr = 4 ** k
        self.n_slots = self.n_addr * self.S    # entry index = addr * S + s
        if write_gate not in ("content", "surprise"):
            raise ValueError(f"write_gate must be content|surprise, got {write_gate!r}")
        self.write_gate_mode = write_gate
        self.write_proj = nn.Linear(d_in, mem_dim, bias=False)
        # gate_init -> read_gate (scalar, needs a head start: see configs); write_gate_init ->
        # write_gate.bias (defaults to gate_init). Re-applied by KmerGPT after GPT's bias sweep.
        self.gate_init = float(gate_init)
        self.write_gate_init = float(gate_init if write_gate_init is None else write_gate_init)
        self.write_gate = nn.Linear(d_in, 1)
        nn.init.constant_(self.write_gate.bias, self.write_gate_init)
        self.read_proj = nn.Linear(mem_dim, d_in, bias=False)
        # Read gate: start as pure KDA and opt into the memory (same convention as HOLA).
        self.read_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.read_gate._no_weight_decay = True
        if write_gate == "surprise":
            # s_t ~ 1 when the slot is empty/stale (write) and ~ 0 when it already holds u_t (skip).
            self.surprise_w = nn.Parameter(torch.tensor(float(surprise_init)))
            self.surprise_w._no_weight_decay = True
        self.write_gate.bias._no_weight_decay = True
        if self.S > 1:
            self.router = nn.Linear(d_in, self.S, bias=False)
        self.aux_loss = None       # set per forward when S > 1; drained by KmerGPT
        self.router_stats = None   # dict of router diagnostics per forward when S > 1

    def empty_table(self, batch: int, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_zeros(batch, self.n_slots, self.mem_dim, dtype=torch.float32)

    # ---- the segmented scan -----------------------------------------------------------------

    def scan(self, addr, gamma, u, table):
        """addr (B,T) long, gamma (B,T) fp32 in (0,1), u (B,T,d) fp32, table (B,V,d) fp32.
        Returns reads (B,T,d) — the slot content *before* the write at each position — and the
        table after all writes."""
        B, T, d = u.shape
        V, C = self.n_slots, self.scan_chunk
        dev = u.device
        valid = addr >= 0
        key = (torch.arange(B, device=dev)[:, None] * V + addr.clamp_min(0)).masked_fill(~valid, -1)
        flat_key, flat_valid = key.reshape(-1), valid.reshape(-1)
        order = torch.argsort(flat_key, stable=True)          # invalid (-1) sort first
        n_inv = int((~flat_valid).sum().item())
        order = order[n_inv:]                                  # valid positions, by (chain, time)
        n = order.numel()
        reads = u.new_zeros(B * T, d)
        if n == 0:
            return reads.view(B, T, d), table

        skey = flat_key[order]
        seg_start = torch.ones(n, dtype=torch.bool, device=dev)
        seg_start[1:] = skey[1:] != skey[:-1]
        seg_id = torch.cumsum(seg_start.long(), 0) - 1        # (n,)
        n_seg = int(seg_id[-1].item()) + 1
        first = torch.nonzero(seg_start).squeeze(1)           # (n_seg,)
        rank = torch.arange(n, device=dev) - first[seg_id]    # position within chain
        piece_in_seg = rank // C
        pos_in_piece = rank % C
        seg_pieces = torch.zeros(n_seg, dtype=torch.long, device=dev)
        seg_pieces.scatter_reduce_(0, seg_id, piece_in_seg + 1, reduce="amax")
        piece_base = torch.cumsum(seg_pieces, 0) - seg_pieces  # first piece id of each chain
        piece_id = piece_base[seg_id] + piece_in_seg
        n_pieces = int(seg_pieces.sum().item())
        max_pieces = int(seg_pieces.max().item())

        # --- pack into (n_pieces, C, ·) ---
        g_s = gamma.reshape(-1)[order]
        u_s = u.reshape(-1, d)[order]
        G = u.new_zeros(n_pieces, C)                           # gamma, 0 on padding (decay 1)
        U = u.new_zeros(n_pieces, C, d)
        G[piece_id, pos_in_piece] = g_s
        U[piece_id, pos_in_piece] = u_s

        # --- intra-piece closed form ---
        log1m = torch.log1p(-G).clamp_min(-1e4)              # (P, C), 0 on padding
        L = torch.cumsum(log1m, dim=1)                          # L_n = sum_{i<=n} log(1-g_i)
        Lx = torch.nn.functional.pad(L, (1, 0))[:, :-1]         # exclusive: sum_{i<n}
        # decay from just after j to just before n: sum_{j<i<n} = Lx_n - L_j, for j < n
        diff = Lx[:, :, None] - L[:, None, :]                  # (P, C(n), C(j))
        tri = torch.ones(C, C, dtype=torch.bool, device=dev).tril(-1)
        D = torch.exp(diff).masked_fill(~tri, 0.0)
        local = torch.einsum("pnj,pjd->pnd", D, G.unsqueeze(-1) * U)   # sum_{j<n} ...
        piece_decay_ex = torch.exp(Lx)                          # (P, C): prod_{i<n}(1-g_i)
        piece_decay = torch.exp(L[:, -1])                       # (P,) whole-piece decay
        piece_input = torch.einsum("pj,pjd->pd", torch.exp(L[:, -1:] - L), G.unsqueeze(-1) * U)

        # --- carry across pieces of a chain ---
        seg_key = skey[first]                                   # (n_seg,) = b*V + addr
        seg_b, seg_addr = seg_key // V, seg_key % V
        carry = table[seg_b, seg_addr]                          # (n_seg, d) initial slot state
        init = u.new_zeros(n_pieces, d)
        for j in range(max_pieces):
            has = seg_pieces > j
            pid = piece_base + j
            pid_h = pid[has]
            init[pid_h] = carry[has]
            carry = torch.where(has[:, None],
                                carry * piece_decay[pid.clamp_max(n_pieces - 1)][:, None]
                                + piece_input[pid.clamp_max(n_pieces - 1)],
                                carry)
        R = init[:, None, :] * piece_decay_ex[:, :, None] + local   # (P, C, d)
        reads[order] = R[piece_id, pos_in_piece]
        new_table = table.index_put((seg_b, seg_addr), carry)  # out-of-place
        return reads.view(B, T, d), new_table

    def _entry_addr(self, addr):
        """(B,T) k-mer address -> (B,T,S) entry index a*S+s, preserving the -1 invalid marker."""
        s_idx = torch.arange(self.S, device=addr.device)
        return torch.where(addr[..., None] >= 0, addr[..., None] * self.S + s_idx, -1)

    def _scan_rw(self, read_addr, write_addr, gamma, u, table, p=None):
        """Read the set at read_addr then write it at write_addr, per position, as one interleaved
        event stream of length 2*S*T through the single-address scan: the S read events of position
        t (gamma 0 -> the EMA leaves the entry unchanged) come first, then its S write events, whose
        gates are gamma_t * p_{t,s}. Returns (read per entry, pre-write content per entry, table),
        the first two shaped (B, T, S, d)."""
        B, T, d = u.shape
        S = self.S
        ra, wa = self._entry_addr(read_addr), self._entry_addr(write_addr)     # (B,T,S)
        zeros_s = u.new_zeros(B, T, S)
        w_gamma = gamma[..., None] if p is None else gamma[..., None] * p
        addr2 = torch.cat([ra, wa], dim=2).reshape(B, 2 * S * T)
        gamma2 = torch.cat([zeros_s, w_gamma], dim=2).reshape(B, 2 * S * T)
        u_s = u[:, :, None, :].expand(B, T, S, d)
        u2 = torch.cat([torch.zeros_like(u_s), u_s], dim=2).reshape(B, 2 * S * T, d)
        ev, table = self.scan(addr2, gamma2, u2, table)
        ev = ev.view(B, T, 2 * S, d)
        return ev[:, :, :S], ev[:, :, S:], table

    def _route(self, h):
        """(B,T,S) routing weights, or None when S == 1 (the single-slot path is untouched).

        `soft`: plain softmax. Measured 2026-08-19 on the first d05c run (step 8412): the router
        stayed near-uniform (normalised entropy 0.985), so every entry received gamma*p_s ~= gamma/S
        and the read averaged S identical diluted copies -- i.e. the set degenerated to ONE slot
        with a 4x slower EMA, and needle copy_advantage was 0.08 vs 0.54 for the d05 control.
        `top1_ste`: one-hot argmax in the forward (full gamma into one entry, read from that entry,
        true S-way associativity) with the softmax's gradient straight-through, so the router still
        learns. The soft p is kept for the balance loss / diagnostics."""
        if self.S == 1:
            return None
        p = torch.softmax(self.router(h).float() / self.router_temp, dim=-1)
        self._p_soft = p
        if self.router_mode == "top1_ste":
            hard = F.one_hot(p.argmax(-1), self.S).to(p.dtype)
            p = hard + p - p.detach()
        return p

    def load_balance_loss(self, p):
        """MoE-style: S * sum_s mean(p_s)^2, minimised (== 1) by uniform entry usage. Guards
        against the router collapsing onto one entry, which would silently reduce S back to 1."""
        frac = p.reshape(-1, self.S).mean(0)
        return self.S * (frac * frac).sum()

    @torch.no_grad()
    def _router_stats(self, p):
        """Diagnostics for the S-way router (logged by train.py): mean per-token entropy in
        nats (log S = uniform, 0 = hard routing), normalised by log S; and per-entry occupancy
        fraction = share of tokens whose argmax entry is s. Occupancy max -> 1 means the router
        collapsed and S silently became 1 regardless of what the balance loss says."""
        q = p.reshape(-1, self.S)
        ent = -(q * (q + 1e-9).log()).sum(-1).mean() / math.log(self.S)
        occ = torch.bincount(q.argmax(-1), minlength=self.S).float() / q.shape[0]
        return {"router_entropy": ent.item(), "router_occ_max": occ.max().item(),
                "router_occ_min": occ.min().item()}

    def _surprise(self, logit, u, read_addr, write_addr, table, p):
        """Detached pass-1 scan under gamma_0 -> s_t = ||u_t - r_t^write|| / ||u_t||, where
        r^write is the routed read of the set being written. See docstring for why this is
        two-pass and why pass 1 is detached."""
        with torch.no_grad():
            _, pre, _ = self._scan_rw(read_addr, write_addr, torch.sigmoid(logit).detach(),
                                      u.detach(), table.detach(), p)
            pre = pre.squeeze(2) if p is None else torch.einsum("bts,btsd->btd", p, pre)
        sur = (u.detach() - pre).norm(dim=-1) / (u.detach().norm(dim=-1) + 1e-6)
        return torch.sigmoid(logit + self.surprise_w.float() * sur.clamp_max(10.0))

    def forward(self, h, addr, table):
        """h (B,T,d_in) any dtype; addr = (read_addr, write_addr), each (B,T) long (see module
        docstring; `shift_write_addresses` builds the pair); table (B,V,mem_dim) fp32 or None.
        Returns (contribution to the mixer output (B,T,d_in), new table)."""
        read_addr, write_addr = addr
        B, T, _ = h.shape
        if table is None:
            table = self.empty_table(B, h)
        logit = self.write_gate(h).float().squeeze(-1)
        u = self.write_proj(h).float()
        p = self._route(h)
        p_soft = None if p is None else self._p_soft
        self.aux_loss = None if p is None else self.load_balance_loss(p_soft)
        self.router_stats = None if p is None else self._router_stats(p_soft)
        if self.write_gate_mode == "surprise":
            gamma = self._surprise(logit, u, read_addr, write_addr, table, p)
        else:
            gamma = torch.sigmoid(logit)
        if torch.is_grad_enabled():
            reads, _, table = torch.utils.checkpoint.checkpoint(
                self._scan_rw, read_addr, write_addr, gamma, u, table, p, use_reentrant=False)
        else:
            reads, _, table = self._scan_rw(read_addr, write_addr, gamma, u, table, p)
        reads = reads.squeeze(2) if p is None else torch.einsum("bts,btsd->btd", p, reads)
        out = self.read_proj(reads.to(h.dtype)) * torch.sigmoid(self.read_gate.float()).to(h.dtype)
        return out, table

    def _reference_pass(self, gamma, u, read_addr, write_addr, base, p):
        """Sequential ground truth for one scan pass, over entry indices a*S+s. Returns (routed
        read at read_addr, routed pre-write content at write_addr, final table)."""
        B, T, _ = u.shape
        S = self.S
        reads, pre_writes, slots = [], [], {}
        get = lambda b, e: slots.get((b, e), base[b, e])
        for b in range(B):
            for t in range(T):
                w_t = p[b, t] if p is not None else u.new_ones(1)
                a = int(read_addr[b, t])
                reads.append(sum(w_t[j] * get(b, a * S + j) for j in range(S))
                             if a >= 0 else u.new_zeros(self.mem_dim))
                w = int(write_addr[b, t])
                if w >= 0:
                    pre_writes.append(sum(w_t[j] * get(b, w * S + j) for j in range(S)))
                    for j in range(S):
                        g = gamma[b, t] * w_t[j]
                        slots[(b, w * S + j)] = (1 - g) * get(b, w * S + j) + g * u[b, t]
                else:
                    pre_writes.append(u.new_zeros(self.mem_dim))
        table = base.clone()
        for (b, e), v in slots.items():
            table[b, e] = v.detach()
        shape = (B, T, self.mem_dim)
        return torch.stack(reads).view(shape), torch.stack(pre_writes).view(shape), table

    def reference(self, h, addr, table=None):
        """Sequential per-token loop — ground truth for tests. addr = (read_addr, write_addr)."""
        read_addr, write_addr = addr
        B, T, _ = h.shape
        base = table if table is not None else self.empty_table(B, h)
        logit = self.write_gate(h).float().squeeze(-1)
        u = self.write_proj(h).float()
        p = self._route(h)
        if self.write_gate_mode == "surprise":
            with torch.no_grad():
                _, pre_write, _ = self._reference_pass(
                    torch.sigmoid(logit).detach(), u.detach(), read_addr, write_addr,
                    base.detach(), None if p is None else p.detach())
            sur = (u.detach() - pre_write).norm(dim=-1) / (u.detach().norm(dim=-1) + 1e-6)
            gamma = torch.sigmoid(logit + self.surprise_w.float() * sur.clamp_max(10.0))
        else:
            gamma = torch.sigmoid(logit)
        reads, _, table = self._reference_pass(gamma, u, read_addr, write_addr, base, p)
        out = self.read_proj(reads.to(h.dtype)) * torch.sigmoid(self.read_gate.float()).to(h.dtype)
        return out, table
