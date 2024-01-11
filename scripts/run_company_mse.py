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
    log_path=f"logs/company-bis_{job_id}.log",
    device="cpu",
    cache_path=os.getenv("cache_path"),
    proxy_score_cache=True,
    task="est",
    oracle_budget=2000000,
    max_blocking_ratio=0.2,
    bootstrap_trials=10000,
    log_level="debug",
    output_file="company-bis.jsonl"
)

run(config)
