from joinml.run import run
from joinml.config import Config
import time

job_id = int(time.time())
config = Config(
    seed=job_id,
    dataset_name="VeRi",
    proxy="reid",
    is_self_join=False,
    log_path=f"logs/veri-blocking_{job_id}.log",
    device="cpu",
    cache_path="../.cache/joinml",
    proxy_score_cache=True,
    task="blocking",
    oracle_budget=100000,
    data_path="./data",
    bootstrap_trials=1000,
    log_level="debug",
    output_file="veri-blocking.jsonl",
    internal_loop=100,
)

run(config) 