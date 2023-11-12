from joinml.executable.run_blocking import run
from joinml.config import Config
import logging

config = Config(
    dataset_name="stackoverflow",
    join_algorithm="naive_importance",
    proxy="all-MiniLM-L6-v2",
    is_self_join=True,
    log_path="logs/stackoverflow-blocking_LM.log",
    repeats=20,
    proxy_cache=True,
    device="cpu"
)

run(config)
