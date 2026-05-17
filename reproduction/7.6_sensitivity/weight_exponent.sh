#!/bin/bash
# §7.6 / Figure 13b — sensitivity to the weight exponent γ.
#
# Sweeps γ ∈ {0.5, 1.0, 1.5, 2.0, 3.0, 5.0} on Company at b=2M, internal_loop=100.
# Runs sequentially via scripts/bas_run.sh.

set -euo pipefail
cd "$(dirname "$0")/../.."

INTERNAL_LOOP=${INTERNAL_LOOP:-100}
TAG_ROOT=${TAG_ROOT:-sec7.6-wexp}

GAMMAS=(0.5 1.0 1.5 2.0 3.0 5.0)
cfg=configs/company-minilm.yml
budget=2000000

i=0
for gamma in "${GAMMAS[@]}"; do
  i=$((i + 1))
  echo "==> [$i] γ=$gamma  $cfg budget=$budget"
  W_EXP="$gamma" bash scripts/bas_run.sh "$cfg" "$budget" "$INTERNAL_LOOP" \
      "${TAG_ROOT}-${gamma}"
done
echo "[7.6 γ] Done."
