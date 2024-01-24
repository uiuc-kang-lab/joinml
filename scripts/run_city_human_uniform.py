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
        log_path=f"logs/city_human-uniform_{job_id}.log",
        device="cpu",
        cache_path="../.cache/joinml",
        proxy_score_cache=True,
        task="uniform",
        oracle_budget=oracle_budget,
        max_blocking_ratio=0.2,
        bootstrap_trials=10000,
        log_level="info",
        internal_loop=11,
        output_file="city_human-uniform.jsonl"
    )

    run(config)
