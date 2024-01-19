from joinml.run import run
from joinml.config import Config
import time
import os

for _ in range(100):
    job_id = int(time.time())
    config = Config(
        seed=job_id,
        dataset_name="company",
        proxy="all-MiniLM-L6-v2",
        is_self_join=False,
        log_path=f"logs/company-joinml-mse_{job_id}.log",
        device="cpu",
        cache_path="../.cache/joinml",
        proxy_score_cache=True,
        task="joinml-mse",
        oracle_budget=2000000,
        max_blocking_ratio=0.4,
        bootstrap_trials=1000,
        log_level="info",
        output_file="company-joinml-mse-optimal.jsonl",
        need_ci=True,
    )

    run(config)


for _ in range(100):
    job_id = int(time.time())
    config = Config(
        seed=job_id,
        dataset_name="company",
        proxy="all-MiniLM-L6-v2",
        is_self_join=False,
        log_path=f"logs/company-joinml-ci_{job_id}.log",
        device="cpu",
        cache_path="../.cache/joinml",
        proxy_score_cache=True,
        task="joinml-ci",
        oracle_budget=2000000,
        max_blocking_ratio=0.4,
        bootstrap_trials=1000,
        log_level="info",
        output_file="company-joinml-ci-optimal.jsonl",
        need_ci=True,
    )

    run(config)

