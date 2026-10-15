#!/usr/bin/env bash
# One-shot setup: uv env, .env, background corpora. Models are NOT downloaded here —
# scripts/run_one.sh / run_all.sh fetch them one at a time.
#   bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv sync --inexact --all-extras
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
[ -f .env ] || { cp .env.example .env; echo "created .env — set HUGGING_FACE_TOKEN"; }
RECALL_BENCH_HOME="$PWD" uv run recall-bench prepare dna text
echo "Next: bash scripts/run_one.sh hyenadna-small-32k"
