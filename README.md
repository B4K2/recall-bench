# recall-bench

[![PyPI](https://img.shields.io/pypi/v/recall-bench)](https://pypi.org/project/recall-bench/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/recall-bench)](https://pypi.org/project/recall-bench/)
[![Downloads](https://static.pepy.tech/badge/recall-bench)](https://pepy.tech/project/recall-bench)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Mnemos--DNA-yellow)](https://huggingface.co/B4K2xx/Mnemos-DNA)

Zero-shot long-range **recall** benchmark for genomic language models: how far back can a DNA
LM still use an exact earlier copy of a sequence? Spun off from the Mnemos-DNA repeat-copy probe
([B4K2/Mnemos-DNA](https://github.com/B4K2/Mnemos-DNA), §6.4). Paper source: [`paper/`](paper/).

**Protocol.** A real 256 bp hg38 segment *A* is repeated after *d* bp of held-out genomic
background. `copy_advantage = CE(control) − CE(repeat)` on the identical final copy of *A*
(nats/bp), where the control window has background instead of the first copy. Causal models are
scored by next-token CE, masked LMs by masking the whole final copy. Text LMs run the same
protocol on wikitext-103 as a positive control. 95% bootstrap CIs over trials.

**Protocol options** (defaults reproduce the paper's main results bit-for-bit):

| option | values | measures |
|---|---|---|
| `--variant` | `exact` · `mutate --mut-rate p` · `revcomp` | exact, diverged (substitutions), or inverted (reverse-complement) repeats |
| `--control` | `native` · `foreign` | control piece is the genome that really precedes the gap, or an unrelated segment (both windows then carry an out-of-place piece, removing the short-*d* context-disruption penalty) |

Every option sees the *same* windows for a given (seed, ctx, d), so options compare paired.

## Install

```bash
pip install "recall-bench[hf,data]"
# latest from GitHub: pip install "recall-bench[hf,data] @ git+https://github.com/B4K2/recall-bench"
```

Extras: `hf` (Hugging Face models), `data` (building the backgrounds), `mnemos` (Mnemos-DNA arms,
Caduceus, GLA/DeltaNet; the Mnemos-DNA model code is bundled),
`all`. The core needs only `torch`, `numpy`, `pyyaml`.
Blackwell (sm_120, RTX 50-series) GPUs need the CUDA 12.8 torch wheels
(`pip install torch --index-url https://download.pytorch.org/whl/cu128`); inside this repo `uv sync --all-extras`
pins them.

Data and weights go under `$RECALL_BENCH_HOME` (default `~/.cache/recall_bench`):

```bash
recall-bench prepare dna          # held-out hg38 background (~1 GB download, once)
```

## Benchmark your own model (Python)

```python
import recall_bench as rb

model = rb.CallableModel(
    logits_fn=lambda ids: net(ids),          # LongTensor (1, T) -> logits (1, T, V)
    tokenize=lambda s: [VOCAB[c] for c in s],
    causal=True,                             # masked LMs: causal=False, mask_id=...
    max_ctx=131072,
)
rows = rb.evaluate(model, distances=[64, 1024, 16384, 65536], trials=128,
                   out="results/my-model/copy.json")          # resumable
for r in rows:
    print(r["dist"], r["copy_advantage"], (r["ci_lo"], r["ci_hi"]))

rb.evaluate(model, variant="mutate", mut_rate=0.05)   # diverged repeats
rb.evaluate(model, variant="revcomp")                 # inverted repeats
rb.evaluate(model, control="foreign")                 # stricter control
```

Any Hugging Face checkpoint: `rb.from_hf("org/model", causal=True, max_ctx=8192, align=1)`
(`align=6` for 6-mer tokenizers). A custom background (another genome, other held-out regions):
`rb.DNABackground.from_strings([...])` passed as `background=`.

## Command line

```bash
recall-bench list                                     # registry (recall_bench/models.yaml)
recall-bench run hyenadna-small-32k --fetch           # download if missing -> sweep -> JSON
recall-bench run mnemos-dna --variant mutate --mut-rate 0.05 --distances 64,1024,16384
recall-bench run-hf InstaDeepAI/nucleotide-transformer-v2-50m-multi-species \
    --masked --max-ctx 2048 --align 6
recall-bench report results/                          # table; --variant/--control to filter
```

Results: `results/<model>/copy.json` (one row per point and protocol; finished points are reused,
each trial is checkpointed). Default sweep: d ∈ {64 … 262144} bp, ctx = smallest power of two
that fits, 128 trials.

## Reproducing the paper (this repo)

```bash
bash scripts/setup.sh                          # uv env (cu128 torch), .env, backgrounds
CLEANUP=1 bash scripts/run_all.sh ours baseline   # main sweep, one model at a time
bash scripts/run_protocols.sh                  # foreign control, revcomp, mutation sweep
uv run python paper/make_figures.py            # figures + tables from results/
cd paper/ieee && pdflatex main && bibtex main && pdflatex main && pdflatex main   # or paper/acm
uv run pytest                                  # CPU-only checks (no data or GPU needed)
```

The repo scripts set `RECALL_BENCH_HOME` to the repo, so `data/` and `checkpoints/` live here.
Set `HUGGING_FACE_TOKEN` in `.env`. Caduceus needs `mamba-ssm` + `causal-conv1d` built for
sm_120 (see Mnemos-DNA CLAUDE.md §9); evo2 needs the `evo2` package and is marked optional.

## Models

| group | models |
|---|---|
| ours | Mnemos-DNA D01, D02, D06 (mnemos-dna), D07 (mnemos-dna-c) |
| baseline | HyenaDNA tiny-1k / small-32k / medium-450k / large-1m, Caduceus-Ph/PS 131k |
| other_dna | Nucleotide Transformer v2 50M/500M, DNABERT-2, GENA-LM, GENERator 1.2B, Evo2-1B (optional) |
| text | Pythia-160M, Mamba-130M, RWKV-4-169M, GLA-340M, DeltaNet-1.3B |

Adding a model is one line in `recall_bench/models.yaml`.

## License

MIT, see [LICENSE](LICENSE).
