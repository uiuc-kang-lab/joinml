from joinml.run import run
from joinml.config import Config
import time

config = Config(
    dataset_name="company",
    proxy="all-MiniLM-L6-v2",
    is_self_join=False,
    log_path=f"logs/company-uniform_{time.time()}.log",
    device="cpu",
    cache_path="../.cache/joinml",
    proxy_score_cache=True,
    task="uniform",
    oracle_budget=1000000,
    bootstrap_trials=10000,
    log_level="DEBUG",
    output_file="company-uniform.jsonl"
)

run(config)
