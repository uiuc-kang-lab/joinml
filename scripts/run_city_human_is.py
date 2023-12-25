from joinml.run import run
from joinml.config import Config
import time

for oracle_budget in [20000 * i for i in range(1, 12)]:
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
        num_strata=11,
        max_blocking_ratio=0.2,
        bootstrap_trials=10000,
        log_level="info",
        output_file="city_human-is.jsonl"
    )

    run(config)
