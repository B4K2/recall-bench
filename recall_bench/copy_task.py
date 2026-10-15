"""Zero-shot repeat-copy recall probe (extends Mnemos-DNA §6.4 `--needle-mode copy`).

    repeat : [ lead bg ][ segment A ][ d units of background ][ segment A ]
    control: [ lead bg ][ bg        ][ d units of background ][ segment A ]
                                                               ^^^^^^^^^ scored

copy_advantage = CE(control) - CE(repeat), in nats per unit (bp for DNA, token for text), over
the identical final segment. The two windows differ only in the earlier occurrence, so the
paired diff isolates recall of A across d intervening units. Causal models score the segment
by next-token CE; masked LMs by masking the whole final segment and predicting it in one pass
(their only route to A is then the earlier copy, read bidirectionally).

Units: DNA pieces are strings of bases, tokenized piece-by-piece and concatenated, so the
scored tokens are byte-identical in both windows even under k-mer/BPE tokenizers (piece
lengths are multiples of the model's `align`). Text pieces are token slices directly.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

LEAD_MIN = 64
ACGT = set(b"ACGT")


class DNABackground:
    def __init__(self, path):
        z = np.load(path)
        self.seq, self.off = z["seq"], z["offsets"]

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


def build_pair(adapter, bg, rng, ctx, dist, seg):
    """Returns (repeat_ids, control_ids, span) with span = (start, end) of the scored tokens."""
    lead = ctx - 2 * seg - dist
    if adapter.domain == "dna":
        a = adapter.align
        lead, seg_, dist_ = _round(lead, a), _round(seg, a), _round(dist, a)
        w = bg.sample(rng, lead + 2 * seg_ + dist_)
        p_lead, p_ctrl, p_dist = w[:lead], w[lead:lead + seg_], w[lead + seg_:lead + seg_ + dist_]
        p_seg = w[lead + seg_ + dist_:]
        t = adapter.tokenize
        T_lead, T_ctrl, T_dist, T_seg = t(p_lead), t(p_ctrl), t(p_dist), t(p_seg)
    else:
        w = bg.sample(rng, lead + 3 * seg + dist)
        T_lead, T_ctrl = w[:lead], w[lead:lead + seg]
        T_dist, T_seg = w[lead + seg:lead + seg + dist], w[lead + seg + dist:lead + 2 * seg + dist]
    pre = adapter.prefix
    rep = pre + T_lead + T_seg + T_dist + T_seg + adapter.suffix
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


def run_point(adapter, bg, ctx, dist, seg, trials, seed, partial=(), on_trial=None):
    """partial: per-trial diffs from an interrupted run; their samples are replayed (no model
    call) so the rng stream and final result match an uninterrupted run. on_trial(diffs) is
    called after every new trial so the caller can checkpoint."""
    rng = np.random.default_rng([seed, ctx, dist])
    diffs, ntok = list(partial), 0
    for i in range(trials):
        rep, ctl, sr, sc = build_pair(adapter, bg, rng, ctx, dist, seg)
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
