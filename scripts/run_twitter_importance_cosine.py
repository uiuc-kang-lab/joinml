from joinml.config import Config
from joinml.executable.run_one_proxy import run
import logging

config = Config(
    dataset_name="twitter",
    join_algorithm="naive_importance",
    proxy="Cosine",
    is_self_join=True,
    log_path="logs/twitter-ni-cosine.log",
    repeats=20,
    proxy_cache=True
)

join_sample_sizes = [1000, 10000, 100000, 1000000, 10000000]

for join_sample_size in join_sample_sizes:
    config.join_sample_size = join_sample_size
    logging.info(f"Running {config.join_algorithm} with {config.join_sample_size}...")
    run(config)