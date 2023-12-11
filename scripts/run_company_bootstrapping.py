from joinml.executable.run_bootstrapping import run
from joinml.config import Config
import logging

config = Config(
    dataset_name="company",
    proxy="all-MiniLM-L6-v2",
    is_self_join=False,
    log_path="logs/company-bootstrapping.log",
    device="cpu",
    cache_path="/mydata/yuxuan",
    proxy_score_cache=True,
    num_strata=11,
    oracle_budget=1000000,
    bootstrap_trials=5000
)

run(config)
