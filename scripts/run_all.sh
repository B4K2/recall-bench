#!/usr/bin/env bash
# Models strictly one at a time: download -> test -> (delete) -> next. Failures are logged and
# skipped so one broken model doesn't stop the queue.
#   bash scripts/run_all.sh                      # every non-optional model, keeps weights
#   CLEANUP=1 bash scripts/run_all.sh ours baseline   # only these groups, delete after each
set -uo pipefail
cd "$(dirname "$0")/.."
groups="${*:-}"
names=$(uv run python - "$groups" <<'P'
import sys, yaml
g = sys.argv[1].split()
for n, e in yaml.safe_load(open("models.yaml")).items():
    if (not g or e["group"] in g) and not e.get("optional"):
        print(n)
P
)
mkdir -p results
for n in $names; do
  if bash scripts/run_one.sh "$n" ${CLEANUP:+--cleanup} 2>&1 | tee "results/$n.log"; then :
  else echo "FAILED: $n (see results/$n.log)"; fi
done
