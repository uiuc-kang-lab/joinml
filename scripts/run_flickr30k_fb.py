from joinml.run import run
from joinml.config import Config
import time
import os

for blocking_ratio in [0.1, 0.2, 0.3, 0.4]:
    job_id = int(time.time())
    config = Config(
        seed=job_id,
        dataset_name="flickr30k",
        proxy="blip",
        is_self_join=False,
        log_path=f"logs/flickr30k-fb_{job_id}.log",
        device="cpu",
        cache_path="../.cache/joinml",
        proxy_score_cache=True,
        task="fb",
        oracle_budget=2000000,
        bootstrap_trials=1000,
        blocking_ratio=blocking_ratio,
        log_level="debug",
        output_file="flickr30k-fb.jsonl",
        internal_loop=100
    )

    run(config)
