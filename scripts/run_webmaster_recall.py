from joinml.run import run
from joinml.config import Config
import time
job_id = int(time.time())
config = Config(
    seed=job_id,
    dataset_name="webmasters",
    proxy="all-MiniLM-L6-v2",
    is_self_join=True,
    log_path=f"logs/webmaster_recall.log",
    device="cpu",
    cache_path="../.cache/joinml",
    proxy_score_cache=True,
    task="recall",
    log_level="DEBUG",
    output_file="stackoverflow-recall.jsonl"
)

run(config)
