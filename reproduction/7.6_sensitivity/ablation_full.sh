#!/bin/bash
# §7.6 — Broad knob ablation on four primary datasets.
#
# Sweeps the major BaS knobs (allocation_search, defensive_mix_ratio,
# max_blocking_ratio, strata_size, sampling_scheme, w_exp) on
# Company, Quora, VeRi, Ecomm-Q7 at their lowest paper budgets.
#
# This is the wider sensitivity exploration used during reproduction.

set -euo pipefail
cd "$(dirname "$0")/../.."

INTERNAL_LOOP=${INTERNAL_LOOP:-100}

# (config, budget) at the lowest paper budget for each primary
primaries=(
  "configs/company-minilm.yml    1000000"
  "configs/quora-minilm.yml      1000000"
  "configs/veri.yml               100000"
  "configs/ecomm-q7.yml            10000"
)

# (variant_tag, flags)
variants=(
  "baseline       "
  "C1-subset      --allocation_search subset"
  "C3-wexp-0.5    --w_exp 0.5"
  "C3-wexp-1.5    --w_exp 1.5"
  "C3-wexp-2.0    --w_exp 2.0"
  "C3-wexp-3.0    --w_exp 3.0"
  "C4-defmix-001  --defensive_mix_ratio 0.001"
  "C4-defmix-01   --defensive_mix_ratio 0.01"
  "C4-defmix-05   --defensive_mix_ratio 0.05"
  "C5-alpha-10    --max_blocking_ratio 0.10"
  "C5-alpha-15    --max_blocking_ratio 0.15"
  "C5-alpha-25    --max_blocking_ratio 0.25"
  "C5-alpha-30    --max_blocking_ratio 0.30"
  "C6-strata-500  --strata_size 500"
  "C6-strata-2000 --strata_size 2000"
  "C7-wor         --sampling_scheme wor"
)

# 4 primaries × 16 variants = 64 cells (long; intentionally not throttled).
total=$(( ${#primaries[@]} * ${#variants[@]} ))
i=0
for entry in "${primaries[@]}"; do
  read -r cfg budget <<<"$entry"
  for v in "${variants[@]}"; do
    read -r tag rest <<<"$v"
    i=$((i + 1))
    echo "==> [$i/$total] $cfg budget=$budget tag=$tag ${rest:+flags=[$rest]}"
    if [ -n "${rest:-}" ]; then
      bash scripts/bas_run.sh "$cfg" "$budget" "$INTERNAL_LOOP" "$tag" $rest
    else
      bash scripts/bas_run.sh "$cfg" "$budget" "$INTERNAL_LOOP" "$tag"
    fi
  done
done
echo "[7.6 ablation-full] Done."
