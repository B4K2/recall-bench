#!/usr/bin/env bash
# Protocol extensions from the paper (foreign control, inverted and diverged repeats), one model
# at a time: download -> every protocol -> delete weights. Resumable like run_all.sh.
#   bash scripts/run_protocols.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export RECALL_BENCH_HOME="${RECALL_BENCH_HOME:-$PWD}"
D=64,256,1024,4096,16384,65536
MODELS="mnemos-dna mnemos-dna-c pure-kda-d01 kda-local-d02 hyenadna-small-32k hyenadna-medium-450k
        hyenadna-large-1m caduceus-ph-131k caduceus-ps-131k"
MEMORY="mnemos-dna mnemos-dna-c"
for n in $MODELS; do
  {
    echo "==> [$n] protocols"
    uv run recall-bench download "$n" >/dev/null
    uv run recall-bench run "$n" --distances $D --control foreign
    uv run recall-bench run "$n" --distances $D --variant revcomp
    if [[ " $MEMORY " == *" $n "* ]]; then
      for p in 0.01 0.05 0.1 0.2; do
        uv run recall-bench run "$n" --distances 64,1024,16384,65536 --variant mutate --mut-rate $p
      done
    fi
    rm -rf "checkpoints/$n"
  } 2>&1 | grep -v -E "Warning|deprecated" | tee -a "results/$n.protocols.log"
done
