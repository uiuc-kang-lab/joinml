from joinml.run import run
from joinml.config import Config
import time

for oracle_budget in [4000000, 5000000, 6000000, 7000000, 8000000]:
    job_id = int(time.time())
    config = Config(
        seed=job_id,
        dataset_name="webmasters",
        proxy="all-MiniLM-L6-v2",
        is_self_join=True,
        log_path=f"logs/webmaster_uniform_{job_id}.log",
        device="cpu",
        cache_path="../.cache/joinml",
        proxy_score_cache=True,
        task="uniform",
        oracle_budget=oracle_budget,
        log_level="DEBUG",
        max_blocking_ratio=0.2,
        output_file="webmaster-uniform.jsonl",
        internal_loop=100
    )

    run(config)
