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
    log_path=f"logs/company-blocking_{job_id}.log",
    device="cpu",
    cache_path=os.getenv("cache_path"),
    proxy_score_cache=True,
    task="blocking",
    oracle_budget=1000000,
    log_level="DEBUG",
    output_file="company-blocking.jsonl",
    internal_loop=100,
)

run(config)
