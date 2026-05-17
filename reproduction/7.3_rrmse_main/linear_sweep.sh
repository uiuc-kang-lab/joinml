#!/bin/bash
# §7.3 / Figure 7a–7h — linear-aggregator RRMSE sweep.
#
# Runs each (dataset, budget) cell sequentially via scripts/bas_run.sh.
# Per-dataset flags reflect the best-known configuration identified during
#
# Override:
#   INTERNAL_LOOP=200 bash reproduction/7.3_rrmse_main/linear_sweep.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

INTERNAL_LOOP=${INTERNAL_LOOP:-200}
TAG=${TAG:-sec7.3-linear}

# (config, budget, dataset-specific flags)
jobs=(
  # Company (COUNT)
  "configs/company-minilm.yml      1000000   "
  "configs/company-minilm.yml      1500000   "
  "configs/company-minilm.yml      2000000   "
  "configs/company-minilm.yml      2500000   "
  "configs/company-minilm.yml      3000000   "
  # Quora (AVG) — force-block-when-concentrated
  "configs/quora-minilm.yml        1000000   --force_block_concentrated True"
  "configs/quora-minilm.yml        2000000   --force_block_concentrated True"
  "configs/quora-minilm.yml        3000000   --force_block_concentrated True"
  "configs/quora-minilm.yml        4000000   --force_block_concentrated True"
  "configs/quora-minilm.yml        5000000   --force_block_concentrated True"
  # Webmasters (SUM)
  "configs/webmaster-minilm.yml    5000000   "
  "configs/webmaster-minilm.yml    6000000   "
  "configs/webmaster-minilm.yml    7000000   "
  "configs/webmaster-minilm.yml    8000000   "
  "configs/webmaster-minilm.yml    9000000   "
  "configs/webmaster-minilm.yml   10000000   "
  # VeRi (AVG)
  "configs/veri.yml                 100000   "
  "configs/veri.yml                 200000   "
  "configs/veri.yml                 300000   "
  "configs/veri.yml                 400000   "
  "configs/veri.yml                 500000   "
  # Roxford (COUNT)
  "configs/roxford.yml             1000000   "
  "configs/roxford.yml             2000000   "
  "configs/roxford.yml             3000000   "
  "configs/roxford.yml             4000000   "
  "configs/roxford.yml             5000000   "
  # Flickr30k (COUNT)
  "configs/flickr30k.yml           1000000   "
  "configs/flickr30k.yml           2000000   "
  "configs/flickr30k.yml           3000000   "
  "configs/flickr30k.yml           4000000   "
  "configs/flickr30k.yml           5000000   "
  # Ecomm-Q7 (COUNT)
  "configs/ecomm-q7.yml              10000   "
  "configs/ecomm-q7.yml              20000   "
  "configs/ecomm-q7.yml              30000   "
  "configs/ecomm-q7.yml              40000   "
  "configs/ecomm-q7.yml              50000   "
  # Movie-Q5 (AVG) — defensive mixture
  "configs/movie-q5.yml               1000   --defensive_mix_ratio 0.01"
  "configs/movie-q5.yml               2000   --defensive_mix_ratio 0.01"
  "configs/movie-q5.yml               3000   --defensive_mix_ratio 0.01"
  "configs/movie-q5.yml               4000   --defensive_mix_ratio 0.01"
  "configs/movie-q5.yml               5000   --defensive_mix_ratio 0.01"
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
echo "[7.3 linear] Done."
