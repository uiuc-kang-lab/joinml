from joinml.executable.run_two_way_sampling import run
from joinml.config import Config
import logging

config = Config(
    dataset_name="quora",
    join_algorithm="naive_importance",
    proxy="Cosine",
    is_self_join=True,
    log_path="logs/quora-blocked_sampling_tune.log",
    repeats=20,
    proxy_cache=True,
    device="cpu",
    blocking_budget=100000,
    blocking_budgets=[100*i for i in range(1,41)]
)

run(config)
