#!/bin/bash
# End-to-end smoke test of the install + data + embedding cache + algorithm.
#
# Runs one short `joinml-adapt` job per in-scope dataset at its lowest paper
# budget. Sequential — single-machine, no slurm dependency. Per-dataset flags
# match the best-known configuration (same as reproduction/7.3_rrmse_main/).
#
# internal_loop=50 → roughly 1–10 min per dataset. Results are too noisy to
# tightly match results.py but should be within ~30% of the BaS target.

set -euo pipefail
cd "$(dirname "$0")/../.."

INTERNAL_LOOP=${INTERNAL_LOOP:-50}
TAG=${TAG:-setup-verify}

# (config, lowest-budget, best-known flags) — matches §7.3 sweep configuration
jobs=(
  "configs/company-minilm.yml    1000000  "
  "configs/quora-minilm.yml      1000000  --force_block_concentrated True"
  "configs/webmaster-minilm.yml  5000000  "
  "configs/veri.yml               100000  "
  "configs/roxford.yml           1000000  "
  "configs/flickr30k.yml         1000000  "
  "configs/ecomm-q7.yml            10000  "
  "configs/ecomm-q8.yml          1000000  --allocation_search evt"
  "configs/ecomm-q9.yml            25000  "
  "configs/movie-q5.yml             1000  --defensive_mix_ratio 0.01"
  "configs/movie-q6.yml             1000  --allocation_search evt"
)
# Ecomm-Q10 / Ecomm-Q11 deferred — Kronecker-product proxy caches for the
# `clip` proxy on multi-way joins are not yet wired in joinml/proxy/get_proxy.py.
# (Templates exist for `quora_three`/`city_human_three`.)

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
echo "[verify] Done."
