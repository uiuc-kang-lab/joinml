#!/bin/bash
# §7.2 / Figure 5 — statistical-guarantee experiments.
#
# Four (dataset, aggregator) cells at multiple budgets, internal_loop=500,
# bootstrap-t CIs enabled. Acceptance: `scripts/check_guarantees.py` reports
# 95th-percentile error ratio ≤ 1 for every cell.
#
# Runs sequentially via scripts/bas_run.sh.

set -euo pipefail
cd "$(dirname "$0")/../.."

INTERNAL_LOOP=${INTERNAL_LOOP:-500}
TAG=${TAG:-sec7.2}

# (config, list-of-budgets) — Figure 5 cells
jobs=(
  "configs/roxford.yml         500000 700000 900000"               # Fig 5a — Roxford COUNT
  "configs/quora-minilm.yml    4000000 6000000 8000000"            # Fig 5b — Quora AVG
  "configs/webmaster-minilm.yml 4000000 6000000 8000000"           # Fig 5c — Webmasters SUM
  "configs/ecomm-q9.yml        25000 50000 75000 100000 125000"    # Fig 5d — Ecomm-Q9 MEDIAN
)

i=0
for entry in "${jobs[@]}"; do
  set -- $entry
  cfg=$1; shift
  for budget in "$@"; do
    i=$((i + 1))
    echo "==> [$i] $cfg budget=$budget"
    # Quora gets the AVG-best flag, same as the §7.3 sweep
    if [[ "$cfg" == *quora* ]]; then
      bash scripts/bas_run.sh "$cfg" "$budget" "$INTERNAL_LOOP" "$TAG" \
          --force_block_concentrated True
    else
      bash scripts/bas_run.sh "$cfg" "$budget" "$INTERNAL_LOOP" "$TAG"
    fi
  done
done
echo "[7.2] Done. Evaluate with: for f in runs/$TAG/*.jsonl; do uv run python scripts/check_guarantees.py \$f; done"
