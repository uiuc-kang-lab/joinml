#!/bin/bash
# §7.6 — Focused knob ablation on the Phase-D outlier cells:
#   Quora 3M / 4M  (AVG, allocator-instability)
#   Movie-Q5 1k / 3k / 5k  (AVG, small-budget noise)
#
# 16 cells total. Sequential via scripts/bas_run.sh.

set -euo pipefail
cd "$(dirname "$0")/../.."

INTERNAL_LOOP=${INTERNAL_LOOP:-200}

# (config, budget, variant_tag, flags)
jobs=(
  "configs/quora-minilm.yml      1000000  fix-rerun        "
  "configs/quora-minilm.yml      3000000  C-defmix-001     --defensive_mix_ratio 0.01"
  "configs/quora-minilm.yml      4000000  C-defmix-001     --defensive_mix_ratio 0.01"
  "configs/quora-minilm.yml      3000000  C-subset         --allocation_search subset"
  "configs/quora-minilm.yml      4000000  C-subset         --allocation_search subset"
  "configs/quora-minilm.yml      3000000  C-defmix-subset  --defensive_mix_ratio 0.01 --allocation_search subset"
  "configs/quora-minilm.yml      4000000  C-defmix-subset  --defensive_mix_ratio 0.01 --allocation_search subset"
  "configs/movie-q5.yml             1000  C-defmix-001     --defensive_mix_ratio 0.01"
  "configs/movie-q5.yml             3000  C-defmix-001     --defensive_mix_ratio 0.01"
  "configs/movie-q5.yml             5000  C-defmix-001     --defensive_mix_ratio 0.01"
  "configs/movie-q5.yml             1000  C-subset         --allocation_search subset"
  "configs/movie-q5.yml             3000  C-subset         --allocation_search subset"
  "configs/movie-q5.yml             5000  C-subset         --allocation_search subset"
  "configs/movie-q5.yml             1000  C-strata-500     --strata_size 500"
  "configs/movie-q5.yml             3000  C-strata-500     --strata_size 500"
  "configs/movie-q5.yml             5000  C-strata-500     --strata_size 500"
)

total=${#jobs[@]}
i=0
for entry in "${jobs[@]}"; do
  i=$((i + 1))
  read -r CFG BUDGET TAG REST <<<"$entry"
  echo "==> [$i/$total] $CFG budget=$BUDGET tag=$TAG ${REST:+flags=[$REST]}"
  if [ -n "$REST" ]; then
    bash scripts/bas_run.sh "$CFG" "$BUDGET" "$INTERNAL_LOOP" "$TAG" $REST
  else
    bash scripts/bas_run.sh "$CFG" "$BUDGET" "$INTERNAL_LOOP" "$TAG"
  fi
done
echo "[7.6 ablation-focused] Done."
