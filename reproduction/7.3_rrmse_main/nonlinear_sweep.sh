#!/bin/bash
# §7.3 / Figure 7i–7k — non-linear-aggregator RRMSE sweep.
#
# Runs each cell sequentially via scripts/bas_run.sh. 
#
# Override:
#   INTERNAL_LOOP=200 bash reproduction/7.3_rrmse_main/nonlinear_sweep.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

INTERNAL_LOOP=${INTERNAL_LOOP:-200}
TAG=${TAG:-sec7.3-nonlinear}

jobs=(
  # Ecomm-Q8 (MIN) — GEV-fit allocator + aggregator override
  "configs/ecomm-q8.yml          1000000  --aggregator min --allocation_search evt"
  "configs/ecomm-q8.yml          2000000  --aggregator min --allocation_search evt"
  "configs/ecomm-q8.yml          3000000  --aggregator min --allocation_search evt"
  "configs/ecomm-q8.yml          4000000  --aggregator min --allocation_search evt"
  "configs/ecomm-q8.yml          5000000  --aggregator min --allocation_search evt"
  # Ecomm-Q9 (MEDIAN) — defaults
  "configs/ecomm-q9.yml            25000  "
  "configs/ecomm-q9.yml            50000  "
  "configs/ecomm-q9.yml            75000  "
  "configs/ecomm-q9.yml           100000  "
  "configs/ecomm-q9.yml           125000  "
  # Movie-Q6 (MAX) — GEV-fit allocator
  "configs/movie-q6.yml             1000  --allocation_search evt"
  "configs/movie-q6.yml             2000  --allocation_search evt"
  "configs/movie-q6.yml             3000  --allocation_search evt"
  "configs/movie-q6.yml             4000  --allocation_search evt"
  "configs/movie-q6.yml             5000  --allocation_search evt"
)

total=${#jobs[@]}
i=0
for entry in "${jobs[@]}"; do
  i=$((i + 1))
  read -r CFG BUDGET REST <<<"$entry"
  echo "==> [$i/$total] $CFG budget=$BUDGET ${REST:+flags=[$REST]}"
  if [ -n "$REST" ]; then
    bash scripts/bas_run.sh "$CFG" "$BUDGET" "$INTERNAL_LOOP" "$TAG" $REST
  else
    bash scripts/bas_run.sh "$CFG" "$BUDGET" "$INTERNAL_LOOP" "$TAG"
  fi
done
echo "[7.3 nonlinear] Done."
