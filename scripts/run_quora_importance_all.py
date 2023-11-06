from joinml.config import Config
from joinml.executable.run_dataset_all_proxy import run
import logging

config = Config(
    dataset_name="quora",
    join_algorithm="naive_importance",
    is_self_join=True,
    log_path="logs/quora-ni-all.log",
    repeats=20,
    proxy_cache=True,
    join_sample_size=1000000
)

join_sample_sizes = [1000, 10000, 100000, 1000000, 10000000]

logging.info(f"Running {config.join_algorithm} with {config.join_sample_size}...")
run(config)