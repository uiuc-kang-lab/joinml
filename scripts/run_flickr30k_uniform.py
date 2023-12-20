from joinml.run import run
from joinml.config import Config
import time

config = Config(
    dataset_name="flickr30k",
    proxy="blip",
    is_self_join=False,
    log_path=f"logs/flickr30k-uniform_{time.time()}.log",
    device="cpu",
    cache_path="../.cache/joinml",
    proxy_score_cache=True,
    task="uniform",
    oracle_budget=1000000,
    num_strata=11,
    max_blocking_ratio=0.2,
    bootstrap_trials=10000,
    log_level="DEBUG",
    output_file="flickr30k-uniform.jsonl"
)

run(config)
