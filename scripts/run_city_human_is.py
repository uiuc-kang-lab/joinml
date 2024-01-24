from joinml.run import run
from joinml.config import Config
import time

for oracle_budget in [20000 * i for i in range(3, 8)]:
    job_id = int(time.time())
    config = Config(
        seed=job_id,
        dataset_name="city_human",
        proxy="human_reid",
        is_self_join=False,
        log_path=f"logs/city_human-is_{job_id}.log",
        device="cpu",
        cache_path="../.cache/joinml",
        proxy_score_cache=True,
        task="is",
        oracle_budget=oracle_budget,
        max_blocking_ratio=0.2,
        bootstrap_trials=1000,
        internal_loop=14,
        log_level="info",
        output_file="city_human-fb.jsonl"
    )

    run(config)
