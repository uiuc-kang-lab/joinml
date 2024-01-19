from joinml.run import run
from joinml.config import Config
import time, os

for oracle_budget in reversed([1000000, 2000000]):
    job_id = int(time.time())
    config = Config(
        dataset_name="stackoverflow",
        proxy="all-MiniLM-L6-v2",
        is_self_join=True,
        log_path=f"logs/stackoverflow-joinml-ci_{job_id}.log",
        device="cpu",
        cache_path="../.cache/joinml",
        proxy_score_cache=True,
        task="joinml-ci",
        oracle_budget=oracle_budget,
        max_blocking_ratio=0.2,
        bootstrap_trials=1000,
        log_level="info",
        output_file="stackoverflow-joinml-ci.jsonl",
        need_ci=True,
        seed=job_id,
        internal_loop=100,
    )

    run(config)
