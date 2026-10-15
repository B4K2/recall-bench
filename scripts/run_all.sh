#!/usr/bin/env bash
# Models strictly one at a time: download -> test -> (delete) -> next. Failures are logged and
# skipped so one broken model doesn't stop the queue.
#   bash scripts/run_all.sh                      # every non-optional model, keeps weights
#   CLEANUP=1 bash scripts/run_all.sh ours baseline   # only these groups, delete after each
# Stop / continue:
#   touch results/STOP        # graceful: finish the current model, then exit
#   Ctrl-C                    # immediate: abort the queue (partial points are kept)
#   rerun the same command    # continues: finished models (results/<n>/.done) are skipped and
#                             # a half-finished model resumes from its saved distances
#   FORCE=1 ...               # ignore .done markers and revisit every model
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
rm -f results/STOP
trap 'echo; echo "interrupted - rerun the same command to continue"; exit 130' INT TERM
for n in $names; do
  if [ -e results/STOP ]; then rm -f results/STOP; echo "STOP file found - halting before $n"; exit 0; fi
  if [ -e "results/$n/.done" ] && [ -z "${FORCE:-}" ]; then echo "skip $n (done)"; continue; fi
  if bash scripts/run_one.sh "$n" ${CLEANUP:+--cleanup} 2>&1 | tee -a "results/$n.log"; then
    mkdir -p "results/$n"; touch "results/$n/.done"
  else echo "FAILED: $n (see results/$n.log)"; fi
done
