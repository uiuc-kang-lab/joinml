from joinml.executable.run_blocked_importance_sampling import run
from joinml.config import Config
import logging

config = Config(
    dataset_name="quora",
    proxy="all-MiniLM-L6-v2",
    is_self_join=True,
    log_path="logs/quora-blocking_sampling.log",
    repeats=50,
    device="cpu",
    cache_path="/mydata/yuxuan",
    proxy_score_cache=True,
    sample_size=[5000000, 2500000, 1000000, 750000, 500000, 250000, 100000],
    blocking_size=[100000, 50000, 10000]
)

run(config)
