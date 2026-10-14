# recall-bench

Zero-shot long-range **recall** benchmark for genomic language models: how far back can a DNA
LM still use an exact earlier copy of a sequence? Spun off from the Mnemos-DNA repeat-copy probe
([B4K2/Mnemos-DNA](https://github.com/B4K2/Mnemos-DNA), §6.4).

**Protocol.** A real 256 bp hg38 segment *A* is repeated after *d* bp of held-out genomic
background. `copy_advantage = CE(control) − CE(repeat)` on the identical final copy of *A*
(nats/bp), where the control window has background instead of the first copy. Causal models are
scored by next-token CE, masked LMs by masking the whole final copy. Text LMs run the same
protocol on wikitext-103 as a positive control. 95% bootstrap CIs over trials.

## Setup

```bash
bash scripts/setup.sh          # uv env (cu128 torch), .env, hg38 test intervals + wikitext
```

Set `HUGGING_FACE_TOKEN` in `.env`. Caduceus needs `mamba-ssm` + `causal-conv1d` built for
sm_120 (see Mnemos-DNA CLAUDE.md §9); evo2 needs the `evo2` package and is marked optional.

## Running — one model at a time

Models are downloaded **only when tested**, one at a time:

```bash
uv run python scripts/download_model.py --list            # registry (models.yaml)
bash scripts/run_one.sh hyenadna-large-1m                  # download -> test
bash scripts/run_one.sh nt-v2-500m --cleanup               # ... and delete weights after
CLEANUP=1 bash scripts/run_all.sh ours baseline            # sequential queue over groups
```

Results: `results/<model>/copy.json` (resumable; finished points are reused).
Default sweep: d ∈ {64 … 262144} bp, ctx = smallest power of two that fits, 128 trials.

## Models

| group | models |
|---|---|
| ours | Mnemos-DNA D01, D02, D06 (mnemos-dna), D07 (mnemos-dna-c) |
| baseline | HyenaDNA tiny-1k / small-32k / medium-450k / large-1m, Caduceus-Ph/PS 131k |
| other_dna | Nucleotide Transformer v2 50M/500M, DNABERT-2, GENA-LM, GENERator 1.2B, Evo2-1B (optional) |
| text | Pythia-160M, Mamba-130M, RWKV-4-169M, GLA-340M, DeltaNet-1.3B |
