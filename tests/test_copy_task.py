"""CPU-only checks of the probe's invariants and of its two reference models: one that cannot
recall (uniform) must score exactly zero; one that copies (k-gram induction) must score high."""
import json

import numpy as np
import pytest
import torch

import recall_bench as rb
from recall_bench.cli import main as cli
from recall_bench.copy_task import build_pair, mutate, revcomp

VOCAB = {c: i for i, c in enumerate("ACGTN")}


def encode(s):
    return [VOCAB[c] for c in s]


@pytest.fixture(scope="module")
def bg():
    rng = np.random.default_rng(0)
    return rb.DNABackground.from_strings(["".join(rng.choice(list("ACGT"), 20000)) for _ in range(3)])


def uniform(ids):
    return torch.zeros(1, ids.shape[1], len(VOCAB))


def induction(ids, k=12):
    """Predict the base that followed the most recent earlier occurrence of the current k-mer."""
    x = ids[0].tolist()
    out = torch.zeros(1, len(x), len(VOCAB))
    seen = {}
    for t in range(len(x)):
        if t >= k - 1:
            key = tuple(x[t - k + 1:t + 1])
            if key in seen and seen[key] + 1 < len(x):
                out[0, t, x[seen[key] + 1]] = 10.0
            seen[key] = t
    return out


def model(fn, **kw):
    return rb.CallableModel(fn, encode, causal=True, max_ctx=1 << 16, device="cpu", **kw)


def test_scored_segment_identical_and_options_share_windows(bg):
    m = model(uniform)
    pairs = {}
    for opts in [{}, {"variant": "mutate", "mut_rate": 0.1}, {"variant": "revcomp"},
                 {"control": "foreign"}]:
        rng, aux = np.random.default_rng([0, 2048, 256]), np.random.default_rng([0, 2048, 256, 1])
        rep, ctl, sr, sc = build_pair(m, bg, rng, 2048, 256, 256, aux=aux, **opts)
        assert rep[sr[0]:sr[1]] == ctl[sc[0]:sc[1]]               # identical scored segment
        assert rep[sr[0] - 256:sr[0]] == ctl[sc[0] - 256:sc[0]]   # identical d bp before it
        pairs[opts.get("variant", "") + opts.get("control", "")] = (rep, ctl)
    rep0, ctl0 = pairs[""]
    assert pairs["revcomp"][1] == ctl0 and pairs["mutate"][1] == ctl0   # same control window
    assert pairs["foreign"][0] == rep0                                  # same repeat window
    assert pairs["foreign"][1] != ctl0


def test_default_options_do_not_touch_aux_stream(bg):
    """Exact/native must reproduce the original benchmark windows bit-for-bit."""
    m = model(uniform)
    a = build_pair(m, bg, np.random.default_rng(1), 2048, 64, 256)
    b = build_pair(m, bg, np.random.default_rng(1), 2048, 64, 256, aux=np.random.default_rng(9))
    assert a == b


def test_mutate_and_revcomp():
    assert revcomp("AACGTN") == "NACGTT"
    s = "ACGT" * 2500
    m = mutate(s, 0.1, np.random.default_rng(0))
    assert len(m) == len(s) and 0.08 < np.mean([a != b for a, b in zip(s, m)]) < 0.12


def test_uniform_model_scores_exactly_zero(bg):
    rows = rb.evaluate(model(uniform), [64, 1024], trials=4, background=bg, verbose=False)
    assert all(r["status"] == "ok" and r["copy_advantage"] == 0 for r in rows)


def test_induction_model_recalls_exact_not_revcomp(bg):
    kw = dict(trials=4, background=bg, verbose=False)
    exact = rb.evaluate(model(induction), [1024], **kw)[0]["copy_advantage"]
    mut = rb.evaluate(model(induction), [1024], variant="mutate", mut_rate=0.05, **kw)[0]
    rc = rb.evaluate(model(induction), [1024], variant="revcomp", **kw)[0]["copy_advantage"]
    foreign = rb.evaluate(model(induction), [1024], control="foreign", **kw)[0]["copy_advantage"]
    assert exact > 1.0                       # near-perfect copying of 256 bp
    assert 0 < mut["copy_advantage"] < exact  # substitutions break some k-mer matches
    assert abs(rc) < 0.05                    # forward induction cannot use an inverted copy
    assert foreign == pytest.approx(exact, abs=0.05)


def test_skip_above_max_ctx(bg):
    m = rb.CallableModel(uniform, encode, max_ctx=1024, device="cpu")
    assert rb.evaluate(m, [64], trials=2, background=bg, verbose=False)[0]["status"].startswith("skip")


def test_resume_and_report(bg, tmp_path, capsys):
    out = tmp_path / "toy" / "copy.json"
    m = model(induction, name="toy")
    first = rb.evaluate(m, [64], trials=3, background=bg, out=out, verbose=False)
    again = rb.evaluate(m, [64], trials=3, background=bg, out=out, verbose=False)
    assert first == again and len(json.loads(out.read_text())["rows"]) == 1
    cli(["report", str(tmp_path)])
    assert "toy" in capsys.readouterr().out
