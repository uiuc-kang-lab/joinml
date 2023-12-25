from joinml.run import run
from joinml.config import Config
import time

for oracle_budget in [1000000 * i for i in range(1, 11)]:
    job_id = int(time.time())
    config = Config(
        seed=job_id,
        dataset_name="flickr30k",
        proxy="blip",
        is_self_join=False,
        log_path=f"logs/flickr30k-bis_{job_id}.log",
        device="cpu",
        cache_path="../.cache/joinml",
        proxy_score_cache=True,
        task="bis",
        oracle_budget=oracle_budget,
        num_strata=11,
        max_blocking_ratio=0.2,
        bootstrap_trials=10000,
        log_level="DEBUG",
        output_file="flickr30k-bis.jsonl"
    )

    run(config)
