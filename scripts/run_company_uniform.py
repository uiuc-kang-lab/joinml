from joinml.run import run
from joinml.config import Config
import time
import os

for oracle_budget in [1000000 * i for i in range(11)]:
    job_id = int(time.time())
    config = Config(
        seed=job_id,
        dataset_name="company",
        proxy="all-MiniLM-L6-v2",
        is_self_join=False,
        log_path=f"logs/company-uniform_{job_id}.log",
        device="cpu",
        cache_path=os.getenv("cache_path"),
        proxy_score_cache=True,
        task="uniform",
        oracle_budget=oracle_budget,
        bootstrap_trials=10000,
        log_level="debug",
        output_file="company-uniform.jsonl"
    )

    run(config)
