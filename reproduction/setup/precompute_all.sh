#!/bin/bash
# Precompute proxy-score caches for every dataset used in the preprint.
#
# Runs `precompute_embeddings.py` once per (dataset, proxy) pair, reading the
# config files in configs/. The cached .npy files are large (Quora ~28 GB,
# Flickr30k ~38 GB) — make sure the cache_path filesystem has room.
#
# This must complete before any of the §7.* experiments will be able to run.

set -euo pipefail
cd "$(dirname "$0")/../.."

CACHE_PATH="${CACHE_PATH:-../.cache/joinml}"
mkdir -p "$CACHE_PATH"

# (config, proxy) pairs used in the preprint. Skip ones whose data isn't local.
configs=(
  configs/company-minilm.yml
  configs/quora-minilm.yml
  configs/webmaster-minilm.yml
  configs/veri.yml
  configs/roxford.yml
  configs/flickr30k.yml
  configs/ecomm-q7.yml
  configs/ecomm-q8.yml
  configs/ecomm-q9.yml
  configs/movie-q5.yml
  configs/movie-q6.yml
)

for cfg in "${configs[@]}"; do
  if [ ! -f "$cfg" ]; then
    echo "[skip] $cfg (config missing)"
    continue
  fi
  ds_name=$(grep '^dataset_name' "$cfg" | awk '{print $2}' | tr -d '"')
  if [ ! -d "data/$ds_name" ]; then
    echo "[skip] $cfg — data/$ds_name/ missing (download first)"
    continue
  fi
  echo "[precompute] $cfg → $CACHE_PATH/"
  uv run python reproduction/setup/precompute_embeddings.py \
      --dataset_config "$cfg" \
      --cache_path "$CACHE_PATH" \
      || echo "  (failed for $cfg; continuing)"
done
echo "Done."
