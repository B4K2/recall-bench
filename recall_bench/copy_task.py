"""Zero-shot repeat-copy recall probe (extends Mnemos-DNA §6.4 `--needle-mode copy`).

    repeat : [ lead bg ][ segment A' ][ d units of background ][ segment A ]
    control: [ lead bg ][ C          ][ d units of background ][ segment A ]
                                                                ^^^^^^^^^ scored

copy_advantage = CE(control) - CE(repeat), in nats per unit (bp for DNA, token for text), over
the identical final segment. The two windows differ only in the earlier piece, so the paired
diff isolates recall of A across d intervening units. Causal models score the segment by
next-token CE; masked LMs by masking the whole final segment and predicting it in one pass
(their only route to A is then the earlier copy, read bidirectionally).

Protocol options (defaults reproduce the original benchmark bit-for-bit):
  variant  what the earlier piece A' is
           exact    A' = A
           mutate   A with each base substituted w.p. `mut_rate` (diverged repeats)
           revcomp  reverse complement of A (inverted repeats; DNA only)
  control  what the control piece C is
           native   the genome that really precedes D (C is untouched background)
           foreign  a segment from an unrelated location, so the repeat and control windows
                    both carry an out-of-place piece and differ only in its *identity*
                    (removes the short-distance context-disruption penalty of `native`)

Units: DNA pieces are strings of bases, tokenized piece-by-piece and concatenated, so the
scored tokens are byte-identical in both windows even under k-mer/BPE tokenizers (piece
lengths are multiples of the model's `align`). Text pieces are token slices directly.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

LEAD_MIN = 64
VARIANTS = ("exact", "mutate", "revcomp")
CONTROLS = ("native", "foreign")
_COMP = str.maketrans("ACGTN", "TGCAN")
_SUB = {b: [c for c in "ACGT" if c != b] for b in "ACGT"}


class DNABackground:
    def __init__(self, path=None, *, seq=None, offsets=None):
        if path is not None:
            z = np.load(path)
            seq, offsets = z["seq"], z["offsets"]
        self.seq, self.off = np.asarray(seq, dtype=np.uint8), np.asarray(offsets)

    @classmethod
    def from_strings(cls, blocks):
        """Background from in-memory sequences (one contiguous block each), e.g. another genome."""
        arrs = [np.frombuffer(b.upper().encode(), dtype=np.uint8) for b in blocks]
        return cls(seq=np.concatenate(arrs), offsets=np.cumsum([0] + [len(a) for a in arrs]))

    def sample(self, rng, n: int) -> str:
        """n contiguous bp with <1% N, from one held-out block (several random blocks
        concatenated if no block is long enough — recorded by the caller)."""
        lens = np.diff(self.off)
        ok = np.flatnonzero(lens >= n)
        for _ in range(200):
            if len(ok):
                b = rng.choice(ok)
                s = rng.integers(self.off[b], self.off[b + 1] - n + 1)
                w = self.seq[s:s + n]
            else:
                parts, got = [], 0
                while got < n:
                    b = rng.integers(len(lens)); parts.append(self.seq[self.off[b]:self.off[b + 1]])
                    got += len(parts[-1])
                w = np.concatenate(parts)[:n]
            if (w == ord("N")).mean() < 0.01:
                return w.tobytes().decode()
        raise RuntimeError("could not find an N-free background window")

    def contiguous(self, n):
        return bool((np.diff(self.off) >= n).any())


class TextBackground:
    def __init__(self, path, adapter):
        self.ids = np.asarray(adapter.tokenize(open(path).read()), dtype=np.int64)

    def sample(self, rng, n):
        s = rng.integers(0, len(self.ids) - n)
        return self.ids[s:s + n].tolist()


def _round(n, a):
    return max(a, (n // a) * a)


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def mutate(s: str, rate: float, rng) -> str:
    """Independent substitutions at `rate` per base (N is left alone)."""
    hit = rng.random(len(s)) < rate
    pick = rng.integers(0, 3, len(s))
    return "".join(_SUB[c][k] if h and c in _SUB else c for c, h, k in zip(s, hit, pick))


def build_pair(adapter, bg, rng, ctx, dist, seg, variant="exact", mut_rate=0.0, control="native",
               aux=None):
    """Returns (repeat_ids, control_ids, repeat_span, control_span); spans are (start, end) of
    the scored tokens. Randomness for non-default options comes from the separate `aux`
    stream, so `rng` -- and hence every lead/background/segment -- is identical across options
    for a given (seed, ctx, dist), and the default options reproduce the original windows."""
    aux = rng if aux is None else aux
    if variant not in VARIANTS or control not in CONTROLS:
        raise ValueError(f"variant must be one of {VARIANTS}, control one of {CONTROLS}")
    lead = ctx - 2 * seg - dist
    if adapter.domain == "dna":
        a = adapter.align
        lead, seg_, dist_ = _round(lead, a), _round(seg, a), _round(dist, a)
        w = bg.sample(rng, lead + 2 * seg_ + dist_)
        p_lead, p_ctrl, p_dist = w[:lead], w[lead:lead + seg_], w[lead + seg_:lead + seg_ + dist_]
        p_seg = w[lead + seg_ + dist_:]
        p_first = p_seg
        if variant == "mutate":
            p_first = mutate(p_seg, mut_rate, aux)
        elif variant == "revcomp":
            p_first = revcomp(p_seg)
        if control == "foreign":
            p_ctrl = bg.sample(aux, seg_)
        t = adapter.tokenize
        T_lead, T_ctrl, T_dist, T_seg = t(p_lead), t(p_ctrl), t(p_dist), t(p_seg)
        T_first = T_seg if p_first is p_seg else t(p_first)
    else:
        if variant != "exact":
            raise ValueError("mutate/revcomp variants are defined for DNA only")
        w = bg.sample(rng, lead + 3 * seg + dist)
        T_lead, T_ctrl = w[:lead], w[lead:lead + seg]
        T_dist, T_seg = w[lead + seg:lead + seg + dist], w[lead + seg + dist:lead + 2 * seg + dist]
        if control == "foreign":
            T_ctrl = bg.sample(aux, seg)
        T_first = T_seg
    pre = adapter.prefix
    rep = pre + T_lead + T_first + T_dist + T_seg + adapter.suffix
    ctl = pre + T_lead + T_ctrl + T_dist + T_seg + adapter.suffix
    # BPE can give the control piece a different token count, so spans are taken from the end.
    end_r = len(rep) - len(adapter.suffix)
    end_c = len(ctl) - len(adapter.suffix)
    return rep, ctl, (end_r - len(T_seg), end_r), (end_c - len(T_seg), end_c)


def segment_ce(adapter, ids, span) -> float:
    """Summed CE (nats) over the tokens in span[0]:span[1]."""
    x = torch.tensor([ids], device=adapter.device)
    s, e = span
    tgt = x[0, s:e]
    if adapter.causal:
        lg = adapter.logits(x)[0, s - 1:e - 1]
    else:
        x = x.clone(); x[0, s:e] = adapter.mask_id
        lg = adapter.logits(x)[0, s:e]
    return F.cross_entropy(lg, tgt, reduction="sum").item()


def run_point(adapter, bg, ctx, dist, seg, trials, seed, partial=(), on_trial=None,
              variant="exact", mut_rate=0.0, control="native"):
    """partial: per-trial diffs from an interrupted run; their samples are replayed (no model
    call) so the rng stream and final result match an uninterrupted run. on_trial(diffs) is
    called after every new trial so the caller can checkpoint."""
    rng = np.random.default_rng([seed, ctx, dist])
    aux = np.random.default_rng([seed, ctx, dist, 1])
    diffs, ntok = list(partial), 0
    for i in range(trials):
        rep, ctl, sr, sc = build_pair(adapter, bg, rng, ctx, dist, seg, variant, mut_rate, control, aux)
        if len(rep) > adapter.max_ctx or len(ctl) > adapter.max_ctx:
            return {"status": "skip: above model max_ctx", "tokens": max(len(rep), len(ctl))}
        ntok = len(rep)
        if i < len(partial):
            continue
        diffs.append((segment_ce(adapter, ctl, sc) - segment_ce(adapter, rep, sr)) / seg)
        if on_trial:
            on_trial(diffs)
    d = np.asarray(diffs)
    boot = np.random.default_rng(seed).choice(d, (1000, len(d))).mean(1)
    return {"status": "ok", "tokens": ntok, "copy_advantage": float(d.mean()),
            "stderr": float(d.std(ddof=1) / math.sqrt(len(d))),
            "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
            "n": len(d)}
