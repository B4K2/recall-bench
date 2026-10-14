#!/usr/bin/env bash
# Download ONE model, run the recall test on it, optionally delete its weights.
#   bash scripts/run_one.sh hyenadna-small-32k [--cleanup] [extra run_copy.py args...]
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1; shift
cleanup=0; args=()
for x in "$@"; do [ "$x" = --cleanup ] && cleanup=1 || args+=("$x"); done
echo "==> [$name] download"
uv run python scripts/download_model.py "$name"
echo "==> [$name] copy-recall probe"
uv run python scripts/run_copy.py "$name" "${args[@]}"
if [ "$cleanup" = 1 ]; then echo "==> [$name] removing checkpoints/$name"; rm -rf "checkpoints/$name"; fi
