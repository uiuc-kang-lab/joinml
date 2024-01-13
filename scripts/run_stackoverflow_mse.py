from joinml.run import run
from joinml.config import Config
import time
import os

for oracle_budget in [1000000 * i for i in range(1, 6)]:
    job_id = int(time.time())
    config = Config(
        seed=job_id,
        dataset_name="stackoverflow",
        proxy="all-MiniLM-L6-v2",
        is_self_join=True,
        log_path=f"logs/stackoverflow-mse_{job_id}.log",
        device="cpu",
        cache_path=os.getenv("cache_path"),
        proxy_score_cache=True,
        task="mse",
        oracle_budget=oracle_budget,
        max_blocking_ratio=0.2,
        log_level="debug",
        output_file="stackoverflow-mse.jsonl"
    )

    run(config)
