from joinml.run import run
from joinml.config import Config
import time

for oracle_budget in [1000000, 2000000, 3000000, 4000000, 5000000]:
    for _ in range(100):
        job_id = int(time.time())
        config = Config(
            dataset_name="twitter",
            proxy="all-MiniLM-L6-v2",
            is_self_join=True,
            log_path=f"logs/twitter-joinml-mse_{job_id}.log",
            device="cpu",
            cache_path="../.cache/joinml",
            proxy_score_cache=True,
            task="joinml-mse",
            oracle_budget=oracle_budget,
            max_blocking_ratio=0.2,
            bootstrap_trials=1000,
            log_level="info",
            output_file="twitter-joinml-mse.jsonl",
            seed=job_id,
            need_ci=True,
            internal_loop=100,
        )

        run(config)
