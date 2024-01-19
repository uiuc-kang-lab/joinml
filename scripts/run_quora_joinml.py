from joinml.run import run
from joinml.config import Config
import time, os

job_id = int(time.time())
config = Config(
    seed=job_id,
    dataset_name="quora",
    proxy="all-MiniLM-L6-v2",
    is_self_join=True,
    log_path=f"logs/quora-joinml_{job_id}.log",
    device="cpu",
    cache_path="../.cache/joinml",
    proxy_score_cache=True,
    task="joinml",
    oracle_budget=4000000,
    max_blocking_ratio=0.2,
    bootstrap_trials=1000,
    log_level="debug",
    output_file="quora-joinml.jsonl",
    internal_loop=100,
    aggregator="count"
)

run(config)
