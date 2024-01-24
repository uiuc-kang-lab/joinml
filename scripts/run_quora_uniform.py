from joinml.run import run
from joinml.config import Config
import time
for oracle_budget in [6000000, 7000000, 8000000]:
    job_id = int(time.time())
    config = Config(
        dataset_name="quora",
        proxy="all-MiniLM-L6-v2",
        is_self_join=True,
        log_path=f"logs/quora-uniform_{job_id}.log",
        device="cpu",
        cache_path="../.cache/joinml",
        proxy_score_cache=True,
        task="uniform",
        oracle_budget=oracle_budget,
        bootstrap_trials=1000,
        log_level="info",
        output_file="quora-uniform.jsonl",
        seed=job_id,
        internal_loop=100
    )
    run(config)
