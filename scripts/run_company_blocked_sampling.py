from joinml.executable.run_blocked_importance_sampling import run
from joinml.config import Config
import logging

config = Config(
    dataset_name="company",
    proxy="all-MiniLM-L6-v2",
    is_self_join=False,
    log_path="logs/company-blocking_sampling-sqrt.log",
    repeats=20,
    device="cpu",
    cache_path="/mydata/yuxuan",
    proxy_score_cache=True,
    sample_size=[5000000, 2500000, 1000000, 750000, 500000, 250000, 100000],
    blocking_size=[1000000, 500000, 100000, 50000, 10000],
    defensive_rate=0,
    proxy_normalizing_style="sqrt"
)

run(config)
