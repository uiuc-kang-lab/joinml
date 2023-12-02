from joinml.executable.run_blocked_importance_sampling import run
from joinml.config import Config
import logging

config = Config(
    dataset_name="city_vehicle_2",
    proxy="reid",
    is_self_join=False,
    log_path="logs/city_vehicle-blocking_sampling-oracle_budget.log",
    repeats=50,
    device="cpu",
    cache_path="/mydata/yuxuan",
    proxy_score_cache=True,
    oracle_budget=2500000
)

run(config)
