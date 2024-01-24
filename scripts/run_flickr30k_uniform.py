from joinml.run import run
from joinml.config import Config
import time
for oracle_budget in [1000000 * i for i in range(2,6)]:
    job_id = int(time.time())
    config = Config(
        seed=job_id,
        dataset_name="flickr30k",
        proxy="blip",
        is_self_join=False,
        log_path=f"logs/flickr30k-uniform_{job_id}.log",
        device="cpu",
        cache_path="../.cache/joinml",
        proxy_score_cache=True,
        task="uniform",
        oracle_budget=oracle_budget,
        max_blocking_ratio=0.2,
        bootstrap_trials=1000,
        internal_loop=5,
        log_level="info",
        output_file="flickr30k-uniform.jsonl"
    )

    run(config)

job_id = int(time.time())
config = Config(
    seed=job_id,
    dataset_name="flickr30k",
    proxy="blip",
    is_self_join=False,
    log_path=f"logs/flickr30k-uniform_{job_id}.log",
    device="cpu",
    cache_path="../.cache/joinml",
    proxy_score_cache=True,
    task="uniform",
    oracle_budget=6000000,
    max_blocking_ratio=0.2,
    bootstrap_trials=1000,
    internal_loop=68,
    log_level="info",
    output_file="flickr30k-uniform.jsonl"
)

run(config)
