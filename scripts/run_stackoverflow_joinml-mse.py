from joinml.run import run
from joinml.config import Config
import time, os

for oracle_budget in [1000000, 2000000, 3000000, 4000000, 5000000]:
    for _ in range(100):
        job_id = int(time.time())
        config = Config(
            seed=job_id,
            dataset_name="stackoverflow",
            proxy="all-MiniLM-L6-v2",
            is_self_join=True,
            log_path=f"logs/stackoverflow-joinml-mse_{job_id}.log",
            device="cpu",
            cache_path=os.getenv("cache_path"),
            proxy_score_cache=True,
            task="joinml-mse",
            oracle_budget=oracle_budget,
            max_blocking_ratio=0.2,
            bootstrap_trials=1000,
            log_level="info",
            output_file="stackoverflow-joinml-mse.jsonl",
            need_ci=True,
        )

        run(config)
