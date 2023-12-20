from joinml.run import run
from joinml.config import Config
import time

config = Config(
    dataset_name="stackoverflow",
    proxy="all-MiniLM-L6-v2",
    is_self_join=True,
    log_path=f"logs/stackoverflow-uniform_{time.time()}.log",
    device="cpu",
    cache_path="../.cache/joinml",
    proxy_score_cache=True,
    task="uniform",
    oracle_budget=1000000,
    num_strata=6,
    max_blocking_ratio=0.2,
    bootstrap_trials=10000,
    log_level="DEBUG",
    output_file="stackoverflow-uniform.jsonl",
    seed=int(time.time())
)

run(config)
