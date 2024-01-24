from joinml.run import run
from joinml.config import Config
import time

for oracle_budget in [60000]:
    job_id = int(time.time())
    config = Config(
        seed=job_id,
        dataset_name="city_human",
        proxy="human_reid",
        is_self_join=False,
        log_path=f"logs/city_human-joinml_{job_id}.log",
        device="cpu",
        cache_path="../.cache/joinml",
        proxy_score_cache=True,
        task="joinml",
        oracle_budget=oracle_budget,
        max_blocking_ratio=0.4,
        bootstrap_trials=1000,
        log_level="debug",
        output_file="city_human-joinml-opt.jsonl",
        internal_loop=100
    )

    run(config)
