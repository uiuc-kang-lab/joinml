from joinml.executable.run_blocking import run
from joinml.config import Config
import logging

config = Config(
    dataset_name="city_vehicle_2",
    join_algorithm="naive_importance",
    proxy="infomin",
    is_self_join=False,
    log_path="logs/city_vehicle-blocking_infomin.log",
    repeats=20,
    proxy_cache=True
)

run(config)
