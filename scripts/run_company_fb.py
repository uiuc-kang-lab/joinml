from joinml.run import run
from joinml.config import Config
import time
import os

for blocking_ratio in [0.1, 0.2, 0.3, 0.4]:
    job_id = int(time.time())
    config = Config(
        seed=job_id,
        dataset_name="company",
        proxy="all-MiniLM-L6-v2",
        is_self_join=False,
        log_path=f"logs/company-blocking_{job_id}.log",
        device="cpu",
        cache_path="../.cache/joinml",
        proxy_score_cache=True,
        task="fb",
        oracle_budget=2000000,
        log_level="DEBUG",
        output_file="company-fb.jsonl",
        blocking_ratio=blocking_ratio,
        internal_loop=100
    )

    run(config)
