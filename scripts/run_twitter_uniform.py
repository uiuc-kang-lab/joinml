from joinml.run import run
from joinml.config import Config
import time

for oracle_budget in [1000000, 2000000, 3000000, 4000000, 5000000]:
    job_id = int(time.time())
    config = Config(
        dataset_name="twitter",
        proxy="all-MiniLM-L6-v2",
        is_self_join=True,
        log_path=f"logs/twitter-uniform_{job_id}.log",
        device="cpu",
        cache_path="../.cache/joinml",
        proxy_score_cache=True,
        task="uniform",
        oracle_budget=oracle_budget,
        bootstrap_trials=1000,
        log_level="info",
        output_file="twitter-uniform.jsonl",
        seed=job_id,
        internal_loop=100
    )

    run(config)
