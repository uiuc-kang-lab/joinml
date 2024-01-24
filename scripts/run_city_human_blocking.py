from joinml.run import run
from joinml.config import Config
import time
import os

job_id = int(time.time())
config = Config(
    seed=job_id,
    dataset_name="city_human",
    proxy="human_reid",
    is_self_join=False,
    log_path=f"logs/city_human-blocking_{job_id}.log",
    device="cpu",
    cache_path="../.cache/joinml",
    proxy_score_cache=True,
    task="blocking",
    oracle_budget=60000,
    log_level="DEBUG",
    output_file="city_human-blocking.jsonl",
    internal_loop=100
)

run(config)
