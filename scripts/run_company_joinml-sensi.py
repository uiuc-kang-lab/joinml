from joinml.run import run
from joinml.config import Config
import time
import os

for max_blocking_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for _ in range(100):
        job_id = int(time.time())
        config = Config(
            seed=job_id,
            dataset_name="company",
            proxy="all-MiniLM-L6-v2",
            is_self_join=False,
            log_path=f"logs/company-joinml-sensi-mse_{job_id}.log",
            device="cpu",
            cache_path="../.cache/joinml",
            proxy_score_cache=True,
            task="joinml-mse",
            oracle_budget=2000000,
            max_blocking_ratio=max_blocking_ratio,
            bootstrap_trials=1000,
            log_level="info",
            output_file="company-joinml-sensi-mse.jsonl",
            need_ci=True,
        )

        run(config)

for max_blocking_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for _ in range(100):
        job_id = int(time.time())
        config = Config(
            seed=job_id,
            dataset_name="company",
            proxy="all-MiniLM-L6-v2",
            is_self_join=False,
            log_path=f"logs/company-joinml-sensi-ci_{job_id}.log",
            device="cpu",
            cache_path="../.cache/joinml",
            proxy_score_cache=True,
            task="joinml-ci",
            oracle_budget=2000000,
            max_blocking_ratio=max_blocking_ratio,
            bootstrap_trials=1000,
            log_level="info",
            output_file="company-joinml-sensi-ci.jsonl",
            need_ci=True,
        )

        run(config)
