#!/bin/bash
# §7.5 / Figure 10 — adaptive vs fixed-α allocation.
#
# For each of {Quora, Company, Roxford, Flickr30K, Webmasters, VeRi}:
#   * fixed-α runs at α ∈ {0.10, 0.20, 0.30, 0.40, 0.50} using joinml-fixed
#   * one adaptive run using joinml-adapt (default α=0.20)
#
# The paper's "Optimal" and "Worst" allocations correspond to the best/worst α
# values from the sweep on each dataset — they're not separate runs.
#
# Runs sequentially via scripts/bas_run.sh. 36 cells total (6 datasets × 6
# variants), internal_loop=100.

set -euo pipefail
cd "$(dirname "$0")/../.."

INTERNAL_LOOP=${INTERNAL_LOOP:-100}
TAG_ROOT=${TAG_ROOT:-sec7.5}

# (config, budget) pairs from Figure 10
configs=(
  "configs/quora-minilm.yml      4000000"
  "configs/company-minilm.yml    2000000"
  "configs/roxford.yml           3000000"
  "configs/flickr30k.yml         3000000"
  "configs/webmaster-minilm.yml  7000000"
  "configs/veri.yml               300000"
)
ALPHAS=(0.10 0.20 0.30 0.40 0.50)

i=0
for entry in "${configs[@]}"; do
  read -r cfg budget <<<"$entry"
  # Fixed-α variants
  for alpha in "${ALPHAS[@]}"; do
    i=$((i + 1))
    echo "==> [$i] fixed α=$alpha  $cfg budget=$budget"
    bash scripts/bas_run.sh "$cfg" "$budget" "$INTERNAL_LOOP" \
        "${TAG_ROOT}-fixed-${alpha}" \
        --task joinml-fixed --max_blocking_ratio "$alpha"
  done
  # Adaptive
  i=$((i + 1))
  echo "==> [$i] adaptive       $cfg budget=$budget"
  bash scripts/bas_run.sh "$cfg" "$budget" "$INTERNAL_LOOP" \
      "${TAG_ROOT}-adapt"
done
echo "[7.5] Done."
