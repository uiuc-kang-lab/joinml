from joinml.run import run
from joinml.config import Config
import time, os

for blocking_ratio in [0.01 * i for i in range(1, 20)] + [0.1* i for i in range(2, 5)]:
    job_id = int(time.time())
    config = Config(
        seed=job_id,
        dataset_name="quora",
        proxy="all-MiniLM-L6-v2",
        is_self_join=True,
        log_path=f"logs/quora-joinml-fix_{job_id}.log",
        device="cpu",
        cache_path="../.cache/joinml",
        proxy_score_cache=True,
        task="fb",
        oracle_budget=2000000,
        blocking_ratio=blocking_ratio,
        bootstrap_trials=1000,
        log_level="info",
        output_file="quora-joinml-fix.jsonl",
        need_ci=True,
        internal_loop=100,
    )

    run(config)
