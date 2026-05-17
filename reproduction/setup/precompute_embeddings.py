"""Precompute proxy similarity scores for a dataset.

Materialises the (|T_1| × |T_2| × …) similarity tensor used by
joinml-adapt's blocking and importance-sampling steps, and stores it under
the proxy-score cache directory expected by `joinml/proxy/get_proxy.py`.

Usage:
    uv run python reproduction/setup/precompute_embeddings.py \
        --dataset_config configs/company-minilm.yml

For every dataset listed in `results.py`:
    bash reproduction/setup/precompute_all.sh

This wraps the same code path used at experiment time, but lets you front-load
the (potentially long) embedding step instead of paying for it at first run.
"""
from __future__ import annotations

import argparse
import os
import sys

# Ensure repo root is on sys.path when invoked directly
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

import yaml  # noqa: E402

from joinml.config import Config
from joinml.dataset_loader import load_dataset
from joinml.proxy.get_proxy import get_proxy_score, get_proxy_rank
from joinml.utils import set_up_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_config", required=True,
                        help="Path to YAML config under configs/")
    parser.add_argument("--data_path", default="data",
                        help="Root directory for datasets (default: data)")
    parser.add_argument("--cache_path", default="../.cache/joinml",
                        help="Where to write the proxy_scores.npy cache "
                             "(default: ../.cache/joinml)")
    parser.add_argument("--log_level", default="info")
    args = parser.parse_args()

    with open(args.dataset_config) as fh:
        cfg_dict = yaml.safe_load(fh)
    cfg_dict.update(dict(
        data_path=args.data_path,
        cache_path=args.cache_path,
        task="recall",  # placeholder; not actually used
        oracle_budget=1,
        max_blocking_ratio=0.2,
        internal_loop=1,
        log_path=f"logs/precompute_{cfg_dict['dataset_name']}.log",
        log_level=args.log_level,
        output_file="/dev/null",
        proxy_score_cache=True,
        target=0.5,
        ci=False,
        w_exp=1.0,
        seed=42,
        table_ids=[0, 1],
        join_reorder=False,
        top_k=5,
        blocking_ratio=0.2,
    ))
    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.cache_path, exist_ok=True)
    config = Config(**cfg_dict)

    set_up_logging(config.log_path, config.log_level)
    dataset = load_dataset(config)

    print(f"[precompute] dataset={config.dataset_name} proxy={config.proxy}")
    scores = get_proxy_score(config, dataset)
    print(f"  scores: shape={scores.shape} dtype={scores.dtype} "
          f"min={scores.min():.4f} max={scores.max():.4f}")
    rank = get_proxy_rank(config, dataset, scores)
    print(f"  rank:   shape={rank.shape} dtype={rank.dtype}")
    print(f"  cached under {args.cache_path}/")


if __name__ == "__main__":
    main()
