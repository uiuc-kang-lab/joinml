from joinml.run import run
from joinml.config import Config
import time, os

for oracle_budget in [20000, 40000, 60000, 80000, 100000]:
    for _ in range(100):
        job_id = int(time.time())
        config = Config(
            seed=job_id,
            dataset_name="city_human",
            proxy="human_reid",
            is_self_join=False,
            log_path=f"logs/city_human-joinml-mse_{job_id}.log",
            device="cpu",
            cache_path=os.getenv("cache_path"),
            proxy_score_cache=True,
            task="joinml-mse",
            oracle_budget=oracle_budget,
            max_blocking_ratio=0.2,
            bootstrap_trials=1000,
            log_level="info",
            output_file="city_human-joinml-mse.jsonl",
            need_ci=True,
        )

        run(config)
