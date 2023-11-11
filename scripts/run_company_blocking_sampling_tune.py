from joinml.executable.run_two_way_sampling import run
from joinml.config import Config
import logging

config = Config(
    dataset_name="company",
    join_algorithm="naive_importance",
    proxy="all-MiniLM-L6-v2",
    is_self_join=False,
    log_path="logs/company-blocking_sampling_tune_100000.log",
    repeats=20,
    proxy_cache=True,
    device="cpu",
    blocking_budget=100000,
    blocking_budgets=[1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000],
    join_sample_size=100000
)

run(config)
