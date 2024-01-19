from joinml.run import run
from joinml.config import Config
import time
import os

job_id = int(time.time())
config = Config(
    seed=job_id,
    dataset_name="company",
    proxy="all-MiniLM-L6-v2",
    is_self_join=False,
    log_path=f"logs/company-joinml{job_id}.log",
    device="cpu",
    cache_path="../.cache/joinml",
    proxy_score_cache=True,
    task="joinml",
    oracle_budget=2000000,
    max_blocking_ratio=0.2,
    bootstrap_trials=1000,
    log_level="debug",
    output_file="company-joinml.jsonl",
    internal_loop=100,
    aggregator="count"
)

run(config)
