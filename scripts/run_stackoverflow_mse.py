from joinml.run import run
from joinml.config import Config
import time
import os

for oracle_budget in [1000000 * i for i in range(2, 6)]:
    for _ in range(100):
        job_id = int(time.time())
        config = Config(
            dataset_name="stackoverflow",
            proxy="all-MiniLM-L6-v2",
            is_self_join=True,
            log_path=f"logs/stackoverflow-mse_{job_id}.log",
            device="cpu",
            cache_path="../.cache/joinml",
            proxy_score_cache=True,
            task="joinml-mse",
            oracle_budget=oracle_budget,
            max_blocking_ratio=0.2,
            bootstrap_trials=1000,
            log_level="info",
            output_file="stackoverflow-mse.jsonl",
            seed=job_id,
            need_ci=True
        )

        run(config)
