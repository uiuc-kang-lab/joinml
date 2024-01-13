from joinml.run import run
from joinml.config import Config
import time
import os

for oracle_budget in [1000000, 2000000, 3000000, 4000000, 5000000]:
    for _ in range(100):
        job_id = int(time.time())
        config = Config(
            seed=job_id,
            dataset_name="flickr30k",
            proxy="blip",
            is_self_join=False,
            log_path=f"logs/flickr30k-joinml-mse_{job_id}.log",
            device="cpu",
            cache_path=os.getenv("cache_path"),
            proxy_score_cache=True,
            task="joinml-mse",
            oracle_budget=oracle_budget,
            max_blocking_ratio=0.2,
            bootstrap_trials=1000,
            log_level="info",
            output_file="flickr30k-joinml-mse.jsonl",
            need_ci=True,
        )

        run(config)
