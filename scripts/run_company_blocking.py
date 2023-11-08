from joinml.executable.run_blocking import run
from joinml.config import Config
import logging

config = Config(
    dataset_name="company",
    join_algorithm="naive_importance",
    proxy="all-MiniLM-L6-v2",
    is_self_join=False,
    log_path="logs/company-blocking_LM.log",
    repeats=20,
    proxy_cache=True
)

run(config)