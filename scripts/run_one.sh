#!/usr/bin/env bash
# Download ONE model, run the recall test on it, optionally delete its weights.
#   bash scripts/run_one.sh hyenadna-small-32k [--cleanup] [extra `recall-bench run` args...]
set -euo pipefail
cd "$(dirname "$0")/.."
export RECALL_BENCH_HOME="${RECALL_BENCH_HOME:-$PWD}"   # repo-local data/ and checkpoints/
name=$1; shift
cleanup=0; args=()
for x in "$@"; do [ "$x" = --cleanup ] && cleanup=1 || args+=("$x"); done
echo "==> [$name] download"
uv run recall-bench download "$name"
echo "==> [$name] copy-recall probe"
uv run recall-bench run "$name" "${args[@]}"
if [ "$cleanup" = 1 ]; then echo "==> [$name] removing checkpoints/$name"; rm -rf "checkpoints/$name"; fi
