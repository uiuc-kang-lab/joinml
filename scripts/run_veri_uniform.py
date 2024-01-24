from joinml.run import run
from joinml.config import Config
import time

for oracle_budget in [100000*i for i in range(2, 6)]:
    job_id = int(time.time())
    config = Config(
        seed=job_id,
        dataset_name="VeRi",
        proxy="reid",
        is_self_join=False,
        log_path=f"logs/veri-uniform_{job_id}.log",
        device="cpu",
        cache_path="../.cache/joinml",
        proxy_score_cache=True,
        task="uniform",
        oracle_budget=oracle_budget,
        data_path="./data",
        bootstrap_trials=1000,
        log_level="debug",
        output_file="veri-uniform_ttest.jsonl",
        internal_loop=100
    )

    run(config) 
