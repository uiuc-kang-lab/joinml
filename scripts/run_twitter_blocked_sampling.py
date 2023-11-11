from joinml.executable.run_blocked_sampling import run
from joinml.config import Config
import logging

config = Config(
    dataset_name="twitter",
    join_algorithm="naive_importance",
    proxy="all-MiniLM-L6-v2",
    is_self_join=True,
    log_path="logs/twitter-blocking_sampling.log",
    repeats=20,
    proxy_cache=True,
    device="cpu",
    blocking_budget=100000
)

run(config)