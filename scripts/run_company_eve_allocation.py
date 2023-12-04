from joinml.executable.run_eves_allocation import run
from joinml.config import Config
import logging

config = Config(
    dataset_name="company",
    proxy="all-MiniLM-L6-v2",
    is_self_join=False,
    log_path="logs/company-eve_allocation.log",
    repeats=50,
    device="cpu",
    cache_path="/mydata/yuxuan",
    proxy_score_cache=True,
    oracle_budget=1000000,
    num_strata=51,
)

run(config)
