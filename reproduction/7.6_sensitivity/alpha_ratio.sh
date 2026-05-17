#!/bin/bash
# §7.6 / Figure 13a — sensitivity to max_blocking_ratio (α).
#
# Sweeps α ∈ {0.10, 0.15, 0.20, 0.25, 0.30} on Webmasters (paper) and
# Flickr30k as a second dataset, at one budget each, internal_loop=100.
#
# Runs sequentially via scripts/bas_run.sh.

set -euo pipefail
cd "$(dirname "$0")/../.."

INTERNAL_LOOP=${INTERNAL_LOOP:-100}
TAG_ROOT=${TAG_ROOT:-sec7.6-alpha}

ALPHAS=(0.10 0.15 0.20 0.25 0.30)
configs=(
  "configs/webmaster-minilm.yml  7000000"
  "configs/flickr30k.yml         3000000"
)

i=0
for entry in "${configs[@]}"; do
  read -r cfg budget <<<"$entry"
  for alpha in "${ALPHAS[@]}"; do
    i=$((i + 1))
    echo "==> [$i] α=$alpha  $cfg budget=$budget"
    bash scripts/bas_run.sh "$cfg" "$budget" "$INTERNAL_LOOP" \
        "${TAG_ROOT}-${alpha}" \
        --max_blocking_ratio "$alpha"
  done
done
echo "[7.6 α] Done."
