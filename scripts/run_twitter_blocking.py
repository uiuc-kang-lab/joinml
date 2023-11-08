from joinml.executable.run_blocking import run
from joinml.config import Config
import logging

config = Config(
    dataset_name="twitter",
    join_algorithm="naive_importance",
    proxy="all-MiniLM-L6-v2",
    is_self_join=True,
    log_path="logs/twitter-blocking_LM.log",
    repeats=20,
    proxy_cache=True
)

run(config)
