from joinml.run import run
from joinml.config import Config
import time
import os

for oracle_budget in [1000000, 3000000, 4000000, 5000000]:

    for _ in range(100):
        job_id = int(time.time())
        config = Config(
            seed=job_id,
            dataset_name="company",
            proxy="all-MiniLM-L6-v2",
            is_self_join=False,
            log_path=f"logs/company-joinml-stats_guarantees_{job_id}.log",
            device="cpu",
            cache_path="../.cache/joinml",
            proxy_score_cache=True,
            task="stats-guarantees",
            oracle_budget=oracle_budget,
            max_blocking_ratio=0.2,
            bootstrap_trials=1000,
            log_level="debug",
            output_file="company-joinml-stats_guarantees.jsonl",
            need_ci=True,
            internal_loop=100
        )

        run(config)
