from joinml.run import run
from joinml.config import Config
import time

config = Config(
    dataset_name="city_human",
    proxy="human_reid",
    is_self_join=False,
    log_path=f"logs/city_human-bis_{time.time()}.log",
    device="cpu",
    cache_path="../.cache/joinml",
    proxy_score_cache=True,
    task="bis",
    oracle_budget=20000,
    num_strata=11,
    max_blocking_ratio=0.2,
    bootstrap_trials=10000,
    log_level="DEBUG",
    output_file="city_human-bis.jsonl"
)

run(config)
